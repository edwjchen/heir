#include "lib/Analysis/ILPBootstrapPlacement/ILPBootstrapPlacement.h"

#include <cstdlib>
#include <sstream>
#include <utility>

#include "lib/Analysis/LevelAnalysis/LevelAnalysis.h"
#include "lib/Analysis/SecretnessAnalysis/SecretnessAnalysis.h"
#include "lib/Dialect/Mgmt/IR/MgmtOps.h"
#include "lib/Dialect/Secret/IR/SecretOps.h"
#include "llvm/include/llvm/ADT/DenseMap.h"       // from @llvm-project
#include "llvm/include/llvm/ADT/DenseSet.h"       // from @llvm-project
#include "llvm/include/llvm/ADT/SmallVector.h"   // from @llvm-project
#include "llvm/include/llvm/ADT/TypeSwitch.h"     // from @llvm-project
#include "llvm/include/llvm/Support/Debug.h"      // from @llvm-project
#include "llvm/include/llvm/Support/raw_ostream.h"  // from @llvm-project
#include "mlir/include/mlir/IR/Block.h"           // from @llvm-project
#include "mlir/include/mlir/IR/Operation.h"       // from @llvm-project
#include "mlir/include/mlir/IR/Value.h"           // from @llvm-project
#include "mlir/include/mlir/Support/LogicalResult.h"  // from @llvm-project

#include "absl/status/statusor.h"  // from @com_google_absl
#include "ortools/math_opt/cpp/math_opt.h"  // from @com_google_ortools

namespace math_opt = ::operations_research::math_opt;

namespace mlir {
namespace heir {

#define DEBUG_TYPE "ilp-bootstrap-placement"

// Big-M constant: large enough to relax constraints when the indicator is off.
static constexpr int kBigM = 1000;

static bool diagnosticsEnabled() {
  bool enabled = std::getenv("HEIR_ILP_DIAGNOSTICS") != nullptr;
  // Force immediate output to verify diagnostics are enabled
  if (enabled) {
    llvm::errs() << "[DIAGNOSTICS CHECK] HEIR_ILP_DIAGNOSTICS is enabled\n";
    llvm::errs().flush();
  }
  return enabled;
}

LogicalResult ILPBootstrapPlacement::solve() {
  if (diagnosticsEnabled()) {
    llvm::errs() << "[ILP solve] Starting solve()\n";
    llvm::errs().flush();
  }
  auto genericOp = dyn_cast<secret::GenericOp>(opToRunOn);
  if (!genericOp) {
    if (diagnosticsEnabled()) {
      llvm::errs() << "[ILP solve] Not a GenericOp, returning early\n";
      llvm::errs().flush();
    }
    LLVM_DEBUG(llvm::dbgs() << "ILP bootstrap: not a secret.generic, skip\n");
    return success();
  }

  Block* body = genericOp.getBody();
  if (!body || body->empty()) {
    if (diagnosticsEnabled()) {
      llvm::errs() << "[ILP solve] Empty body, returning early\n";
      llvm::errs().flush();
    }
    return success();
  }

  math_opt::Model model("ILPBootstrapPlacement");

  // Level-based formulation: level variable per secret value, binary per modreduce.
  llvm::DenseMap<Value, math_opt::Variable> levelVars;
  // Ordered list (execution order) for tie-break: prefer later bootstraps.
  llvm::SmallVector<std::pair<Operation*, math_opt::Variable>, 32> bootstrapVarsOrdered;

  int vid = 0;
  int bid = 0;
  int joinId = 0;

  // Block arguments: secret inputs get level = waterline; non-secret (if any) same.
  // Level = waterline - depth must be integer (depth is number of modreduces).
  for (BlockArgument arg : body->getArguments()) {
    std::string name = "L_arg_" + std::to_string(arg.getArgNumber());
    auto var = model.AddIntegerVariable(0.0, waterline, name);
    levelVars.insert({arg, var});
    model.AddLinearConstraint(var == waterline,
                              "input_level_" + std::to_string(arg.getArgNumber()));
    vid++;
  }

  // Pass 1: create level var for every value (not just secret) so join
  // constraints see all operands and we propagate max correctly; missing
  // operands would under-constrain and force extra bootstraps (e.g. SiLU).
  for (Operation& op : body->getOperations()) {
    if (isa<secret::YieldOp>(&op)) continue;

    llvm::TypeSwitch<Operation*>(&op)
        .Case<mgmt::ModReduceOp>([&](mgmt::ModReduceOp modReduceOp) {
          Value result = modReduceOp.getResult();
          std::string lOutName = "L_out_" + std::to_string(vid++);
          math_opt::Variable lOut =
              model.AddIntegerVariable(0.0, waterline, lOutName);
          levelVars.insert({result, lOut});
        })
        .Default([&](Operation* defaultOp) {
          for (OpResult res : defaultOp->getResults()) {
            std::string name = "L_op_" + std::to_string(vid++);
            math_opt::Variable lRes =
                model.AddIntegerVariable(0.0, waterline, name);
            levelVars.insert({res, lRes});
          }
        });
  }

  // Pass 2: add bootstrap vars and constraints (Big-M at modreduces,
  // L_result = min(operand L's) at joins, since L = waterline - depth).
  int modReduceCount = 0;
  int modReduceSkippedSecret = 0;
  int modReduceSkippedMissing = 0;
  for (Operation& op : body->getOperations()) {
    if (isa<secret::YieldOp>(&op)) continue;

    llvm::TypeSwitch<Operation*>(&op)
        .Case<mgmt::ModReduceOp>([&](mgmt::ModReduceOp modReduceOp) {
          Value input = modReduceOp.getOperand();
          Value result = modReduceOp.getResult();
          modReduceCount++;
          auto itIn = levelVars.find(input);
          auto itOut = levelVars.find(result);
          if (itIn == levelVars.end() || itOut == levelVars.end()) {
            modReduceSkippedMissing++;
            return;
          }
          bool secretResult = isSecret(result, solver);
          if (!secretResult) modReduceSkippedSecret++;

          // Inside secret.generic, add bootstrap var for every modreduce so we
          // can place bootstraps (SecretnessAnalysis may not mark all body
          // values as secret).
          std::string bName = "Bootstrap_" + std::to_string(bid++);
          math_opt::Variable bVar = model.AddBinaryVariable(bName);
          bootstrapVarsOrdered.push_back({modReduceOp.getOperation(), bVar});

          math_opt::Variable lIn = itIn->second;
          math_opt::Variable lOut = itOut->second;

          // b=0: L_out = L_in - 1. b=1: L_out = waterline. Big-M:
          model.AddLinearConstraint(lOut >= lIn - 1 - kBigM * (1 - bVar), "");
          model.AddLinearConstraint(lOut <= lIn - 1 + kBigM * (1 - bVar), "");
          model.AddLinearConstraint(lOut >= waterline - kBigM * (1 - bVar), "");
          model.AddLinearConstraint(lOut <= waterline + kBigM * bVar, "");
          // b=0 => L_in >= 1 so L_out >= 0 (match greedy: insert when level % waterline == 0).
          model.AddLinearConstraint(lIn >= 1 - kBigM * bVar, "");
        })
        .Default([&](Operation* defaultOp) {
          // Include all operands that have level vars for join/min propagation.
          // (Inside secret.generic, block args may not be marked secret by
          // SecretnessAnalysis; excluding them would set L_result=waterline at
          // muls and incorrectly allow 0 bootstraps.)
          llvm::SmallVector<math_opt::Variable, 4> operandLevels;
          for (Value operand : defaultOp->getOperands()) {
            auto it = levelVars.find(operand);
            if (it != levelVars.end()) operandLevels.push_back(it->second);
          }

          for (OpResult result : defaultOp->getResults()) {
            auto itRes = levelVars.find(result);
            if (itRes == levelVars.end()) continue;
            math_opt::Variable lRes = itRes->second;
            if (operandLevels.empty()) {
              // No operands (e.g. constants, inits): full budget like inputs.
              model.AddLinearConstraint(lRes == waterline, "");
            } else if (operandLevels.size() == 1) {
              // Inside secret.generic we always propagate from operands; do not
              // set L = waterline for "non-secret" (SecretnessAnalysis may mark
              // body values non-secret because they have plain tensor type).
              // Exact propagation: L_result = L_operand (match greedy).
              model.AddLinearConstraint(lRes >= operandLevels[0], "");
              model.AddLinearConstraint(lRes <= operandLevels[0], "");
            } else {
              // L = waterline - depth; LevelAnalysis uses depth with join = max(depths).
              // So depth_result = max(depth_operands) => L_result = min(L_operands).
              int k = operandLevels.size();
              llvm::SmallVector<math_opt::Variable, 4> minBinaries;
              for (int i = 0; i < k; ++i) {
                std::string name =
                    "min_join_" + std::to_string(joinId++) + "_" + std::to_string(i);
                minBinaries.push_back(model.AddBinaryVariable(name));
              }
              math_opt::LinearExpression sumB;
              for (math_opt::Variable b : minBinaries) sumB += b;
              model.AddLinearConstraint(sumB == 1, "");
              for (int i = 0; i < k; ++i) {
                model.AddLinearConstraint(
                    lRes >= operandLevels[i] - kBigM * (1 - minBinaries[i]), "");
                model.AddLinearConstraint(
                    lRes <= operandLevels[i] + kBigM * (1 - minBinaries[i]), "");
                for (int j = 0; j < k; ++j)
                  model.AddLinearConstraint(
                      operandLevels[i] <=
                          operandLevels[j] + kBigM * (1 - minBinaries[i]),
                      "");
              }
            }
          }
        });
  }

  if (bootstrapVarsOrdered.empty()) {
    LLVM_DEBUG(llvm::dbgs() << "No ModReduceOps in body, nothing to optimize\n");
    return success();
  }

  if (diagnosticsEnabled()) {
    llvm::errs() << "\n=== ILP Bootstrap Diagnostics ===\n";
    llvm::errs().flush();
    bool greedyFeasible = testGreedySolutionFeasibility();
    bool oneBootstrapFeasible = testBootstrapCountFeasibility(1);
    llvm::errs() << "[ILP bootstrap diagnostics] greedy_feasible="
                 << (greedyFeasible ? "true" : "false")
                 << " exact_1_bootstrap_feasible="
                 << (oneBootstrapFeasible ? "true" : "false") << "\n";
    llvm::errs().flush();
  }

  // Minimize number of bootstraps.
  math_opt::LinearExpression obj;
  for (auto& [op, bVar] : bootstrapVarsOrdered) {
    obj += bVar;
  }
  model.Minimize(obj);

  LLVM_DEBUG({
    std::stringstream ss;
    ss << model;
    llvm::dbgs() << ss.str();
  });

  absl::StatusOr<math_opt::SolveResult> status =
      math_opt::Solve(model, math_opt::SolverType::kGscip);

  if (!status.ok()) {
    llvm::errs() << "Bootstrap placement ILP solve failed: "
                 << status.status().ToString() << "\n";
    return failure();
  }

  const math_opt::SolveResult& result = status.value();
  if (result.termination.reason != math_opt::TerminationReason::kOptimal &&
      result.termination.reason != math_opt::TerminationReason::kFeasible) {
    llvm::errs() << "Bootstrap placement ILP has no feasible solution. "
                 << "Termination: " << (int)result.termination.reason << "\n";
    return failure();
  }

  auto varMap = result.variable_values();
  int numBootstraps = 0;
  for (auto& [op, var] : bootstrapVarsOrdered) {
    bool bootstrap = varMap[var] > 0.5;
    if (bootstrap) numBootstraps++;
    solution.insert({op, bootstrap});
  }

  if (diagnosticsEnabled()) {
    llvm::errs() << "[ILP solve] bootstrap_vars=" << bootstrapVarsOrdered.size()
                 << " solution_bootstraps=" << numBootstraps << "\n";
    for (size_t i = 0; i < bootstrapVarsOrdered.size(); ++i) {
      auto* op = bootstrapVarsOrdered[i].first;
      double bVal = varMap[bootstrapVarsOrdered[i].second];
      llvm::errs() << "  b[" << i << "]=" << bVal
                   << " after " << op->getName() << "\n";
    }
    llvm::errs().flush();
  }

  LLVM_DEBUG({
    llvm::dbgs() << "[ILP bootstrap] modreduces_total=" << modReduceCount
                 << " non_secret=" << modReduceSkippedSecret
                 << " skipped_missing=" << modReduceSkippedMissing
                 << " bootstraps=" << numBootstraps << "\n";
    llvm::dbgs() << "[ILP bootstrap] Exact bootstrap placements:\n";
    for (int i = 0; i < (int)bootstrapVarsOrdered.size(); ++i) {
      auto* op = bootstrapVarsOrdered[i].first;
      auto modReduceOp = cast<mgmt::ModReduceOp>(op);
      bool bootstrap = varMap[bootstrapVarsOrdered[i].second] > 0.5;
      if (bootstrap) {
        Value result = modReduceOp.getResult();
        auto* levelLattice = solver->lookupState<LevelLattice>(result);
        int level = -1;
        if (levelLattice && levelLattice->getValue().isInt()) {
          level = levelLattice->getValue().getInt();
        }
        llvm::dbgs() << "  Position " << i << ": bootstrap after "
                     << op->getName() << " at " << op->getLoc()
                     << ", result=" << result
                     << ", level=" << level << "\n";
      }
    }
  });

  return success();
}

bool ILPBootstrapPlacement::testGreedySolutionFeasibility() {
  auto genericOp = dyn_cast<secret::GenericOp>(opToRunOn);
  if (!genericOp) {
    return true;
  }

  Block* body = genericOp.getBody();
  if (!body || body->empty()) {
    return true;
  }

  // First, determine which modreduces would get bootstraps in greedy solution
  // (when level % waterline == 0)
  llvm::DenseMap<Operation*, bool> greedyBootstrapDecisions;
  llvm::SmallVector<std::pair<Operation*, int>, 32> greedyBootstrapPlacements;
  int greedyBootstrapCount = 0;
  int modReduceIndex = 0;

  for (Operation& op : body->getOperations()) {
    if (auto modReduceOp = dyn_cast<mgmt::ModReduceOp>(&op)) {
      Value result = modReduceOp.getResult();
      if (!isSecret(result, solver)) continue;

      // Get level from LevelAnalysis (this is the level BEFORE bootstrap placement)
      // LevelAnalysis uses 0-to-L (increasing), where 0 is input and L is max depth
      auto* levelLattice = solver->lookupState<LevelLattice>(result);
      if (!levelLattice || !levelLattice->getValue().isInitialized()) {
        LLVM_DEBUG(llvm::dbgs() << "[Greedy feasibility test] No level for modreduce result, "
                                << "LevelAnalysis may not have run. Skipping test.\n");
        // If LevelAnalysis hasn't run, we can't determine greedy solution
        // Return true (assume feasible) to avoid false negatives
        return true;
      }

      auto levelState = levelLattice->getValue();
      if (levelState.isInt()) {
        int level = levelState.getInt();
        // Greedy inserts bootstrap when level % waterline == 0
        // LevelAnalysis gives level from 0 (input) increasing, so level=16 with waterline=16
        // means we've done 16 modreduces, and greedy would bootstrap
        bool shouldBootstrap = (level % waterline == 0);
        greedyBootstrapDecisions[modReduceOp.getOperation()] = shouldBootstrap;
        if (shouldBootstrap) {
          greedyBootstrapCount++;
          greedyBootstrapPlacements.push_back({modReduceOp.getOperation(), level});
          LLVM_DEBUG(llvm::dbgs() << "[Greedy feasibility test] Greedy would bootstrap at "
                                  << "modreduce with LevelAnalysis level " << level << "\n");
        }
        modReduceIndex++;
      } else if (levelState.isMaxLevel()) {
        // MaxLevel means unknown/unbounded - can't determine greedy solution
        LLVM_DEBUG(llvm::dbgs() << "[Greedy feasibility test] MaxLevel encountered, "
                                << "cannot determine greedy solution. Skipping test.\n");
        return true;
      }
    }
  }

  if (greedyBootstrapDecisions.empty()) {
    LLVM_DEBUG(llvm::dbgs() << "No modreduces found for greedy test\n");
    return true;
  }

  if (diagnosticsEnabled()) {
    llvm::errs() << "[Greedy bootstrap] total_bootstraps=" << greedyBootstrapCount
                 << "\n";
    for (auto& [op, level] : greedyBootstrapPlacements) {
      auto modReduceOp = cast<mgmt::ModReduceOp>(op);
      Value result = modReduceOp.getResult();
      llvm::errs() << "  bootstrap_after=" << op->getName()
                   << " result=" << result << " level=" << level << "\n";
    }
  }

  // Now build the ILP model (same as solve())
  math_opt::Model model("ILPBootstrapPlacementGreedyTest");

  llvm::DenseMap<Value, math_opt::Variable> levelVars;
  llvm::DenseMap<Operation*, math_opt::Variable> bootstrapVars;

  int vid = 0;
  int bid = 0;
  int joinId = 0;

  // Block arguments: secret inputs get level = waterline
  for (BlockArgument arg : body->getArguments()) {
    std::string name = "L_arg_" + std::to_string(arg.getArgNumber());
    auto var = model.AddIntegerVariable(0.0, waterline, name);
    levelVars.insert({arg, var});
    model.AddLinearConstraint(var == waterline,
                              "input_level_" + std::to_string(arg.getArgNumber()));
    vid++;
  }

  // Pass 1: create level var for every value
  for (Operation& op : body->getOperations()) {
    if (isa<secret::YieldOp>(&op)) continue;

    llvm::TypeSwitch<Operation*>(&op)
        .Case<mgmt::ModReduceOp>([&](mgmt::ModReduceOp modReduceOp) {
          Value result = modReduceOp.getResult();
          std::string lOutName = "L_out_" + std::to_string(vid++);
          math_opt::Variable lOut =
              model.AddIntegerVariable(0.0, waterline, lOutName);
          levelVars.insert({result, lOut});
        })
        .Default([&](Operation* defaultOp) {
          for (OpResult res : defaultOp->getResults()) {
            std::string name = "L_op_" + std::to_string(vid++);
            math_opt::Variable lRes =
                model.AddIntegerVariable(0.0, waterline, name);
            levelVars.insert({res, lRes});
          }
        });
  }

  // Pass 2: add bootstrap vars and constraints
  for (Operation& op : body->getOperations()) {
    if (isa<secret::YieldOp>(&op)) continue;

    llvm::TypeSwitch<Operation*>(&op)
        .Case<mgmt::ModReduceOp>([&](mgmt::ModReduceOp modReduceOp) {
          Value input = modReduceOp.getOperand();
          Value result = modReduceOp.getResult();
          if (!isSecret(result, solver)) {
            auto itIn = levelVars.find(input);
            auto itOut = levelVars.find(result);
            if (itIn != levelVars.end() && itOut != levelVars.end()) {
              math_opt::Variable lIn = itIn->second;
              math_opt::Variable lOut = itOut->second;
              model.AddLinearConstraint(lOut >= lIn - 1, "");
              model.AddLinearConstraint(lOut <= lIn - 1, "");
              model.AddLinearConstraint(lIn >= 1, "");
            }
            return;
          }

          auto itIn = levelVars.find(input);
          auto itOut = levelVars.find(result);
          if (itIn == levelVars.end() || itOut == levelVars.end()) {
            return;
          }

          std::string bName = "Bootstrap_" + std::to_string(bid++);
          math_opt::Variable bVar = model.AddBinaryVariable(bName);
          bootstrapVars.insert({modReduceOp.getOperation(), bVar});

          math_opt::Variable lIn = itIn->second;
          math_opt::Variable lOut = itOut->second;

          // b=0: L_out = L_in - 1. b=1: L_out = waterline. Big-M:
          model.AddLinearConstraint(lOut >= lIn - 1 - kBigM * (1 - bVar), "");
          model.AddLinearConstraint(lOut <= lIn - 1 + kBigM * (1 - bVar), "");
          model.AddLinearConstraint(lOut >= waterline - kBigM * (1 - bVar), "");
          model.AddLinearConstraint(lOut <= waterline + kBigM * bVar, "");
          model.AddLinearConstraint(lIn >= 1 - kBigM * bVar, "");

          // Fix bootstrap variable to greedy decision
          auto it = greedyBootstrapDecisions.find(modReduceOp.getOperation());
          if (it != greedyBootstrapDecisions.end()) {
            if (it->second) {
              model.AddLinearConstraint(bVar == 1, "greedy_bootstrap");
            } else {
              model.AddLinearConstraint(bVar == 0, "greedy_no_bootstrap");
            }
          }
        })
        .Default([&](Operation* defaultOp) {
          llvm::SmallVector<math_opt::Variable, 4> operandLevels;
          for (Value operand : defaultOp->getOperands()) {
            auto it = levelVars.find(operand);
            if (it != levelVars.end()) operandLevels.push_back(it->second);
          }

          for (OpResult result : defaultOp->getResults()) {
            auto itRes = levelVars.find(result);
            if (itRes == levelVars.end()) continue;
            math_opt::Variable lRes = itRes->second;
            if (operandLevels.empty()) {
              model.AddLinearConstraint(lRes == waterline, "");
            } else if (!isSecret(result, solver)) {
              model.AddLinearConstraint(lRes == waterline, "");
            } else if (operandLevels.size() == 1) {
              model.AddLinearConstraint(lRes >= operandLevels[0], "");
              model.AddLinearConstraint(lRes <= operandLevels[0], "");
            } else {
              // L = waterline - depth => L_result = min(L_operands) at join.
              int k = operandLevels.size();
              llvm::SmallVector<math_opt::Variable, 4> minBinaries;
              for (int i = 0; i < k; ++i) {
                std::string name =
                    "min_join_" + std::to_string(joinId++) + "_" + std::to_string(i);
                minBinaries.push_back(model.AddBinaryVariable(name));
              }
              math_opt::LinearExpression sumB;
              for (math_opt::Variable b : minBinaries) sumB += b;
              model.AddLinearConstraint(sumB == 1, "");
              for (int i = 0; i < k; ++i) {
                model.AddLinearConstraint(
                    lRes >= operandLevels[i] - kBigM * (1 - minBinaries[i]), "");
                model.AddLinearConstraint(
                    lRes <= operandLevels[i] + kBigM * (1 - minBinaries[i]), "");
                for (int j = 0; j < k; ++j)
                  model.AddLinearConstraint(
                      operandLevels[i] <=
                          operandLevels[j] + kBigM * (1 - minBinaries[i]),
                      "");
              }
            }
          }
        });
  }

  // Try to solve (feasibility check, no objective needed)
  absl::StatusOr<math_opt::SolveResult> status =
      math_opt::Solve(model, math_opt::SolverType::kGscip);

  if (!status.ok()) {
    LLVM_DEBUG(llvm::dbgs() << "[Greedy feasibility test] Solve failed: "
                            << status.status().ToString() << "\n");
    return false;
  }

  const math_opt::SolveResult& result = status.value();
  bool isFeasible = (result.termination.reason == math_opt::TerminationReason::kOptimal ||
                     result.termination.reason == math_opt::TerminationReason::kFeasible);

  if (diagnosticsEnabled()) {
    if (isFeasible) {
      llvm::errs() << "[Greedy feasibility test] FEASIBLE\n";
    } else {
      llvm::errs() << "[Greedy feasibility test] INFEASIBLE termination="
                   << (int)result.termination.reason << "\n";
    }
    llvm::errs().flush();
  }

  return isFeasible;
}

void ILPBootstrapPlacement::compareBootstrapPlacements(
    const llvm::SmallVector<std::pair<Operation*, void*>, 32>&
        bootstrapOpsOrdered,
    const llvm::SmallVector<std::pair<Operation*, int>, 32>&
        ilpBootstrapPlacements) {
  if (!diagnosticsEnabled()) return;
  // Get greedy bootstrap placements
  auto genericOp = dyn_cast<secret::GenericOp>(opToRunOn);
  if (!genericOp) return;
  
  Block* body = genericOp.getBody();
  if (!body || body->empty()) return;

  llvm::DenseMap<Operation*, int> greedyPlacements;
  int greedyCount = 0;
  
  for (Operation& op : body->getOperations()) {
    if (auto modReduceOp = dyn_cast<mgmt::ModReduceOp>(&op)) {
      Value result = modReduceOp.getResult();
      if (!isSecret(result, solver)) continue;

      auto* levelLattice = solver->lookupState<LevelLattice>(result);
      if (!levelLattice || !levelLattice->getValue().isInitialized()) continue;

      auto levelState = levelLattice->getValue();
      if (levelState.isInt()) {
        int level = levelState.getInt();
        if (level % waterline == 0) {
          greedyPlacements[modReduceOp.getOperation()] = level;
          greedyCount++;
        }
      }
    }
  }

  // Create sets for easy comparison
  llvm::DenseSet<Operation*> greedyOps;
  llvm::DenseSet<Operation*> ilpOps;
  
  for (auto& [op, level] : greedyPlacements) {
    greedyOps.insert(op);
  }
  
  for (auto& [op, level] : ilpBootstrapPlacements) {
    ilpOps.insert(op);
  }

  llvm::errs() << "Greedy: " << greedyCount << " bootstraps\n";
  llvm::errs() << "ILP: " << ilpBootstrapPlacements.size() << " bootstraps\n\n";

  // Find operations that are in greedy but not ILP
  llvm::SmallVector<Operation*, 32> onlyInGreedy;
  for (auto* op : greedyOps) {
    if (ilpOps.find(op) == ilpOps.end()) {
      onlyInGreedy.push_back(op);
    }
  }

  // Find operations that are in ILP but not greedy
  llvm::SmallVector<Operation*, 32> onlyInILP;
  for (auto* op : ilpOps) {
    if (greedyOps.find(op) == greedyOps.end()) {
      onlyInILP.push_back(op);
    }
  }

  // Find operations in both
  llvm::SmallVector<Operation*, 32> inBoth;
  for (auto* op : greedyOps) {
    if (ilpOps.find(op) != ilpOps.end()) {
      inBoth.push_back(op);
    }
  }

  if (!onlyInGreedy.empty()) {
    llvm::errs() << "Only in Greedy (" << onlyInGreedy.size() << "):\n";
    for (auto* op : onlyInGreedy) {
      auto modReduceOp = cast<mgmt::ModReduceOp>(op);
      Value result = modReduceOp.getResult();
      int level = greedyPlacements[op];
      llvm::errs() << "  Bootstrap after " << op->getName()
                   << ", result=" << result << ", level=" << level << "\n";
    }
  }

  if (!onlyInILP.empty()) {
    llvm::errs() << "Only in ILP (" << onlyInILP.size() << "):\n";
    for (auto* op : onlyInILP) {
      auto modReduceOp = cast<mgmt::ModReduceOp>(op);
      Value result = modReduceOp.getResult();
      int level = -1;
      for (auto& [ilpOp, ilpLevel] : ilpBootstrapPlacements) {
        if (ilpOp == op) {
          level = ilpLevel;
          break;
        }
      }
      llvm::errs() << "  Bootstrap after " << op->getName()
                   << ", result=" << result << ", level=" << level << "\n";
    }
  }

  if (!inBoth.empty()) {
    llvm::errs() << "In both (" << inBoth.size() << "):\n";
    for (auto* op : inBoth) {
      auto modReduceOp = cast<mgmt::ModReduceOp>(op);
      Value result = modReduceOp.getResult();
      int greedyLevel = greedyPlacements[op];
      int ilpLevel = -1;
      for (auto& [ilpOp, lvl] : ilpBootstrapPlacements) {
        if (ilpOp == op) {
          ilpLevel = lvl;
          break;
        }
      }
      llvm::errs() << "  Bootstrap after " << op->getName()
                   << ", result=" << result
                   << ", greedy_level=" << greedyLevel
                   << ", ilp_level=" << ilpLevel << "\n";
    }
  }
  
  llvm::errs() << "\n";
}

bool ILPBootstrapPlacement::testBootstrapCountFeasibility(int targetCount) {
  auto genericOp = dyn_cast<secret::GenericOp>(opToRunOn);
  if (!genericOp) {
    return true;
  }

  Block* body = genericOp.getBody();
  if (!body || body->empty()) return true;

  // Build the ILP model (same as solve())
  math_opt::Model model("ILPBootstrapPlacementCountTest");

  llvm::DenseMap<Value, math_opt::Variable> levelVars;
  llvm::DenseMap<Operation*, math_opt::Variable> bootstrapVars;

  int vid = 0;
  int bid = 0;
  int joinId = 0;

  // Block arguments: secret inputs get level = waterline
  for (BlockArgument arg : body->getArguments()) {
    std::string name = "L_arg_" + std::to_string(arg.getArgNumber());
    auto var = model.AddIntegerVariable(0.0, waterline, name);
    levelVars.insert({arg, var});
    model.AddLinearConstraint(var == waterline,
                              "input_level_" + std::to_string(arg.getArgNumber()));
    vid++;
  }

  // Pass 1: create level var for every value
  for (Operation& op : body->getOperations()) {
    if (isa<secret::YieldOp>(&op)) continue;

    llvm::TypeSwitch<Operation*>(&op)
        .Case<mgmt::ModReduceOp>([&](mgmt::ModReduceOp modReduceOp) {
          Value result = modReduceOp.getResult();
          std::string lOutName = "L_out_" + std::to_string(vid++);
          math_opt::Variable lOut =
              model.AddIntegerVariable(0.0, waterline, lOutName);
          levelVars.insert({result, lOut});
        })
        .Default([&](Operation* defaultOp) {
          for (OpResult res : defaultOp->getResults()) {
            std::string name = "L_op_" + std::to_string(vid++);
            math_opt::Variable lRes =
                model.AddIntegerVariable(0.0, waterline, name);
            levelVars.insert({res, lRes});
          }
        });
  }

  // Pass 2: add bootstrap vars and constraints
  for (Operation& op : body->getOperations()) {
    if (isa<secret::YieldOp>(&op)) continue;

    llvm::TypeSwitch<Operation*>(&op)
        .Case<mgmt::ModReduceOp>([&](mgmt::ModReduceOp modReduceOp) {
          Value input = modReduceOp.getOperand();
          Value result = modReduceOp.getResult();
          if (!isSecret(result, solver)) {
            auto itIn = levelVars.find(input);
            auto itOut = levelVars.find(result);
            if (itIn != levelVars.end() && itOut != levelVars.end()) {
              math_opt::Variable lIn = itIn->second;
              math_opt::Variable lOut = itOut->second;
              model.AddLinearConstraint(lOut >= lIn - 1, "");
              model.AddLinearConstraint(lOut <= lIn - 1, "");
              model.AddLinearConstraint(lIn >= 1, "");
            }
            return;
          }

          auto itIn = levelVars.find(input);
          auto itOut = levelVars.find(result);
          if (itIn == levelVars.end() || itOut == levelVars.end()) {
            return;
          }

          std::string bName = "Bootstrap_" + std::to_string(bid++);
          math_opt::Variable bVar = model.AddBinaryVariable(bName);
          bootstrapVars.insert({modReduceOp.getOperation(), bVar});

          math_opt::Variable lIn = itIn->second;
          math_opt::Variable lOut = itOut->second;

          // b=0: L_out = L_in - 1. b=1: L_out = waterline. Big-M:
          model.AddLinearConstraint(lOut >= lIn - 1 - kBigM * (1 - bVar), "");
          model.AddLinearConstraint(lOut <= lIn - 1 + kBigM * (1 - bVar), "");
          model.AddLinearConstraint(lOut >= waterline - kBigM * (1 - bVar), "");
          model.AddLinearConstraint(lOut <= waterline + kBigM * bVar, "");
          model.AddLinearConstraint(lIn >= 1 - kBigM * bVar, "");
        })
        .Default([&](Operation* defaultOp) {
          llvm::SmallVector<math_opt::Variable, 4> operandLevels;
          for (Value operand : defaultOp->getOperands()) {
            auto it = levelVars.find(operand);
            if (it != levelVars.end()) operandLevels.push_back(it->second);
          }

          for (OpResult result : defaultOp->getResults()) {
            auto itRes = levelVars.find(result);
            if (itRes == levelVars.end()) continue;
            math_opt::Variable lRes = itRes->second;
            if (operandLevels.empty()) {
              model.AddLinearConstraint(lRes == waterline, "");
            } else if (!isSecret(result, solver)) {
              model.AddLinearConstraint(lRes == waterline, "");
            } else if (operandLevels.size() == 1) {
              model.AddLinearConstraint(lRes >= operandLevels[0], "");
              model.AddLinearConstraint(lRes <= operandLevels[0], "");
            } else {
              // L = waterline - depth => L_result = min(L_operands) at join.
              int k = operandLevels.size();
              llvm::SmallVector<math_opt::Variable, 4> minBinaries;
              for (int i = 0; i < k; ++i) {
                std::string name =
                    "min_join_" + std::to_string(joinId++) + "_" + std::to_string(i);
                minBinaries.push_back(model.AddBinaryVariable(name));
              }
              math_opt::LinearExpression sumB;
              for (math_opt::Variable b : minBinaries) sumB += b;
              model.AddLinearConstraint(sumB == 1, "");
              for (int i = 0; i < k; ++i) {
                model.AddLinearConstraint(
                    lRes >= operandLevels[i] - kBigM * (1 - minBinaries[i]), "");
                model.AddLinearConstraint(
                    lRes <= operandLevels[i] + kBigM * (1 - minBinaries[i]), "");
                for (int j = 0; j < k; ++j)
                  model.AddLinearConstraint(
                      operandLevels[i] <=
                          operandLevels[j] + kBigM * (1 - minBinaries[i]),
                      "");
              }
            }
          }
        });
  }

  if (bootstrapVars.empty()) {
    return true;
  }

  // Constraint: exactly targetCount bootstraps
  math_opt::LinearExpression sumB;
  for (auto& [op, bVar] : bootstrapVars) {
    sumB += bVar;
  }
  model.AddLinearConstraint(sumB == targetCount, "exact_bootstrap_count");

  // Try to solve (feasibility check, no objective needed)
  absl::StatusOr<math_opt::SolveResult> status =
      math_opt::Solve(model, math_opt::SolverType::kGscip);

  if (!status.ok()) {
    if (diagnosticsEnabled()) {
      llvm::errs() << "[Bootstrap count test] Solve failed: "
                   << status.status().ToString() << "\n";
    }
    return false;
  }

  const math_opt::SolveResult& result = status.value();
  bool isFeasible = (result.termination.reason == math_opt::TerminationReason::kOptimal ||
                     result.termination.reason == math_opt::TerminationReason::kFeasible);

  if (diagnosticsEnabled()) {
    if (isFeasible) {
      llvm::errs() << "[Bootstrap count test] FEASIBLE: exactly " << targetCount << " bootstrap(s)\n";
    } else {
      llvm::errs() << "[Bootstrap count test] INFEASIBLE: exactly " << targetCount << " bootstrap(s), termination="
                   << (int)result.termination.reason << "\n";
    }
    llvm::errs().flush();
  }

  return isFeasible;
}

}  // namespace heir
}  // namespace mlir
