#include "lib/Analysis/OptimizeBootstrapILPAnalysis/OptimizeBootstrapILPAnalysis.h"

#include <cassert>
#include <sstream>
#include <string>
#include <utility>

#include "lib/Analysis/LevelAnalysis/LevelAnalysis.h"
#include "lib/Analysis/SecretnessAnalysis/SecretnessAnalysis.h"
#include "lib/Dialect/Mgmt/IR/MgmtOps.h"
#include "lib/Dialect/Secret/IR/SecretOps.h"

using ::mlir::heir::getSecretOperands;
using ::mlir::heir::isSecret;
#include "llvm/include/llvm/ADT/DenseMap.h"            // from @llvm-project
#include "llvm/include/llvm/ADT/TypeSwitch.h"          // from @llvm-project
#include "llvm/include/llvm/Support/Casting.h"         // from @llvm-project
#include "llvm/include/llvm/Support/Debug.h"           // from @llvm-project
#include "llvm/include/llvm/Support/raw_ostream.h"     // from @llvm-project
#include "mlir/include/mlir/Dialect/Arith/IR/Arith.h"  // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinOps.h"           // from @llvm-project
#include "mlir/include/mlir/IR/Operation.h"            // from @llvm-project
#include "mlir/include/mlir/IR/Value.h"                // from @llvm-project
#include "mlir/include/mlir/IR/ValueRange.h"           // from @llvm-project
#include "mlir/include/mlir/Support/LLVM.h"            // from @llvm-project
#include "mlir/include/mlir/Support/LogicalResult.h"   // from @llvm-project

// Avoid copybara mangling and separate third party includes with a comment.
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/time/time.h"        // from @com_google_absl
// OR-Tools dependency
#include "ortools/math_opt/cpp/math_opt.h"  // from @com_google_ortools

// The level budget represents the maximum level allowed before bootstrapping
// is required. Levels are tracked from 0 (initial/after bootstrap) upward.
// Operations like modReduce increase the level, and bootstrap resets it to 0.
//
// TODO: Make levelBudget configurable or derive it from scheme parameters.
constexpr int IF_THEN_AUX = 1000;  // Big-M constant for if-then constraints

namespace math_opt = ::operations_research::math_opt;

namespace mlir {
namespace heir {

#define DEBUG_TYPE "optimize-bootstrap-ilp-analysis"

// Helper function to get the level of a value from LevelAnalysis
static std::optional<int> getLevel(Value value, DataFlowSolver* solver) {
  auto* levelLattice = solver->lookupState<LevelLattice>(value);
  if (!levelLattice || !levelLattice->getValue().isInitialized()) {
    return std::nullopt;
  }
  return levelLattice->getValue().getLevel();
}

LogicalResult OptimizeBootstrapILPAnalysis::solve() {
  math_opt::Model model("OptimizeBootstrapILPAnalysis");

  // Generate unique names for variables based on operations
  int nextOpaqueId = 0;
  llvm::DenseMap<Operation*, int> opaqueIds;
  auto uniqueName = [&](Operation* op) {
    std::string varName;
    llvm::raw_string_ostream ss(varName);
    ss << op->getName() << "_";
    if (opaqueIds.count(op) == 0)
      opaqueIds.insert(std::make_pair(op, nextOpaqueId++));

    ss << opaqueIds.lookup(op);
    return ss.str();
  };

  // Map an operation to a decision to bootstrap its results.
  llvm::DenseMap<Operation*, math_opt::Variable> decisionVariables;
  // levelVars maps SSA values to variables tracking the level
  // of the ciphertext at that point in the computation. If the SSA value is
  // the result of an op, this variable corresponds to the level _after_ the
  // decision to bootstrap is applied.
  llvm::DenseMap<Value, math_opt::Variable> levelVars;
  // beforeBootstrapVars is the same as levelVars, but _before_
  // the decision to bootstrap is applied. We need both because the
  // post-processing of the solution requires us to remember the
  // before-bootstrap level.
  llvm::DenseMap<Value, math_opt::Variable> beforeBootstrapVars;

  // First create a variable for each SSA value tracking the level
  // of the ciphertext at that point in the computation, as well as the decision
  // variable to track whether to insert a bootstrap operation after the
  // operation.
  opToRunOn->walk([&](Operation* op) {
    std::string name = uniqueName(op);

    if (isa<ModuleOp>(op)) {
      return;
    }

    // skip secret generic op; we decide inside generic op block
    if (!isa<secret::GenericOp>(op) && isSecret(op->getResults(), solver)) {
      auto decisionVar = model.AddBinaryVariable("InsertBootstrap_" + name);
      decisionVariables.insert(std::make_pair(op, decisionVar));
    }

    // Create level variables for each op result
    std::string varName = "Level_" + name;
    for (OpResult opResult : op->getOpResults()) {
      Value result = opResult;
      varName = varName + "_" + std::to_string(opResult.getResultNumber());

      if (isa<secret::GenericOp>(op) || !isSecret(result, solver)) {
        continue;
      }

      auto levelVar = model.AddContinuousVariable(0, levelBudget, varName);
      levelVars.insert(std::make_pair(result, levelVar));

      // br means "before bootstrap"
      std::string brVarName = varName + "_br";
      auto brLevelVar = model.AddContinuousVariable(0, levelBudget, brVarName);
      beforeBootstrapVars.insert(std::make_pair(result, brLevelVar));
    }

    // Handle block arguments to the op
    if (op->getNumRegions() == 0) {
      return;
    }

    LLVM_DEBUG(llvm::dbgs()
               << "Handling block arguments for " << op->getName() << "\n");
    for (Region& region : op->getRegions()) {
      for (Block& block : region.getBlocks()) {
        for (BlockArgument arg : block.getArguments()) {
          if (!isSecret(arg, solver)) {
            continue;
          }

          std::stringstream ss;
          ss << "Level_ba" << arg.getArgNumber() << "_" << name;
          std::string varName = ss.str();
          auto levelVar = model.AddContinuousVariable(0, levelBudget, varName);
          levelVars.insert(std::make_pair(arg, levelVar));
        }
      }
    }
  });

  // Constraints to initialize the level variables at the start of
  // the computation (block arguments start at level 0).
  for (auto& [value, var] : levelVars) {
    if (llvm::isa<BlockArgument>(value)) {
      auto initialLevel = getLevel(value, solver).value_or(0);
      model.AddLinearConstraint(var == initialLevel, "");
    }
  }

  std::string cstName;
  // Add constraints that set the before_bootstrap variables appropriately
  // based on operation semantics.
  opToRunOn->walk([&](Operation* op) {
    llvm::TypeSwitch<Operation&>(*op)
        .Case<mgmt::ModReduceOp>([&](auto modReduceOp) {
          if (!isSecret(modReduceOp.getResult(), solver)) {
            return;
          }

          auto operandLevelVar = levelVars.at(modReduceOp.getOperand());
          auto resultBeforeBootstrapVar =
              beforeBootstrapVars.at(modReduceOp.getResult());
          std::string opName = uniqueName(op);
          std::string ddPrefix = "BeforeBootstrap_" + opName;

          // before_bootstrap = operand_level + 1 (modReduce increases level)
          cstName = ddPrefix + "_0";
          model.AddLinearConstraint(
              resultBeforeBootstrapVar == operandLevelVar + 1, cstName);
        })
        .Case<mgmt::LevelReduceOp>([&](auto levelReduceOp) {
          if (!isSecret(levelReduceOp.getResult(), solver)) {
            return;
          }

          auto operandLevelVar = levelVars.at(levelReduceOp.getOperand());
          auto resultBeforeBootstrapVar =
              beforeBootstrapVars.at(levelReduceOp.getResult());
          std::string opName = uniqueName(op);
          std::string ddPrefix = "BeforeBootstrap_" + opName;

          // before_bootstrap = operand_level + levelToDrop
          cstName = ddPrefix + "_0";
          model.AddLinearConstraint(
              resultBeforeBootstrapVar ==
                  operandLevelVar + levelReduceOp.getLevelToDrop(),
              cstName);
        })
        .Default([&](Operation& op) {
          // For any other op, the level is the max of input levels
          // (operations typically forward the maximum level of their operands)
          if (isa<secret::GenericOp>(op)) {
            return;
          }
          if (!isSecret(op.getResults(), solver)) {
            return;
          }

          SmallVector<OpOperand*, 4> secretOperandsPtr;
          getSecretOperands(&op, secretOperandsPtr, solver);
          SmallVector<Value, 4> secretOperands;
          for (OpOperand* operand : secretOperandsPtr) {
            if (levelVars.contains(operand->get())) {
              secretOperands.push_back(operand->get());
            }
          }

          if (secretOperands.empty()) {
            return;
          }

          std::string opName = uniqueName(&op);
          std::string ddPrefix = "BeforeBootstrap_" + opName;

          // For mixed-level operations, result level is max of operand levels
          for (OpResult opResult : op.getResults()) {
            Value result = opResult;
            if (!isSecret(result, solver) ||
                !beforeBootstrapVars.contains(result)) {
              continue;
            }

            auto resultBeforeBootstrapVar = beforeBootstrapVars.at(result);

            // before_bootstrap >= each_operand_level
            for (size_t i = 0; i < secretOperands.size(); ++i) {
              Value operand = secretOperands[i];
              auto operandLevelVar = levelVars.at(operand);
              cstName = ddPrefix + "_ge_" + std::to_string(i) + "_" +
                        std::to_string(opResult.getResultNumber());
              model.AddLinearConstraint(
                  resultBeforeBootstrapVar >= operandLevelVar, cstName);
            }
          }
        });
  });

  // Constraint: levels must not exceed the budget
  // If we don't bootstrap, the level must be <= levelBudget
  for (auto& [value, var] : levelVars) {
    if (llvm::isa<BlockArgument>(value)) {
      continue;  // Block arguments are initialized above
    }

    // Find the operation that produces this value
    Operation* definingOp = value.getDefiningOp();
    if (!definingOp || !decisionVariables.contains(definingOp)) {
      // No decision variable for this operation, just constrain level directly
      model.AddLinearConstraint(var <= levelBudget, "");
      continue;
    }

    auto decisionVar = decisionVariables.at(definingOp);
    // If bootstrap is inserted, level can be higher temporarily (but
    // will be reset to 0 by bootstrap). If not inserted, level must be <=
    // budget This is handled by the decision dynamics constraints below
  }

  // The objective is to minimize the number of bootstrap ops.
  // TODO: improve the objective function to account for differing costs
  // of bootstrapping based on the starting level.
  math_opt::LinearExpression obj;
  for (auto& [op, decisionVar] : decisionVariables) {
    obj += decisionVar;
  }
  model.Minimize(obj);

  // Add constraints that control the effect of bootstrap insertion.
  opToRunOn->walk([&](Operation* op) {
    // result_level = before_bootstrap (1 - insert_bootstrap_op) + 0 *
    // insert_bootstrap_op This is linearized using big-M constraints.

    if (isa<secret::GenericOp>(op)) {
      return;
    }
    if (!isSecret(op->getResults(), solver)) {
      return;
    }

    if (!decisionVariables.contains(op)) {
      return;
    }

    for (OpResult opResult : op->getResults()) {
      Value result = opResult;
      if (!levelVars.contains(result) ||
          !beforeBootstrapVars.contains(result)) {
        continue;
      }

      auto resultBeforeBootstrapVar = beforeBootstrapVars.at(result);
      auto resultAfterBootstrapVar = levelVars.at(result);
      auto insertBootstrapDecision = decisionVariables.at(op);
      std::string opName = uniqueName(op);
      std::string ddPrefix = "DecisionDynamics_" + opName + "_" +
                             std::to_string(opResult.getResultNumber());

      // If bootstrap is inserted (decision = 1), result_level = 0
      // If bootstrap is not inserted (decision = 0), result_level =
      // before_bootstrap

      cstName = ddPrefix + "_1";
      // result_level >= 0 (always)
      model.AddLinearConstraint(resultAfterBootstrapVar >= 0, cstName);

      cstName = ddPrefix + "_2";
      // If decision = 1, result_level <= 0 (forces to 0)
      // If decision = 0, result_level can be anything (relaxed by big-M)
      model.AddLinearConstraint(resultAfterBootstrapVar <=
                                    IF_THEN_AUX * (1 - insertBootstrapDecision),
                                cstName);

      cstName = ddPrefix + "_3";
      // If decision = 0, result_level >= before_bootstrap
      // If decision = 1, relaxed by big-M
      model.AddLinearConstraint(
          resultAfterBootstrapVar >=
              resultBeforeBootstrapVar - IF_THEN_AUX * insertBootstrapDecision,
          cstName);

      cstName = ddPrefix + "_4";
      // If decision = 0, result_level <= before_bootstrap
      // If decision = 1, relaxed by big-M
      model.AddLinearConstraint(
          resultAfterBootstrapVar <=
              resultBeforeBootstrapVar + IF_THEN_AUX * insertBootstrapDecision,
          cstName);

      // Constraint: if bootstrap is not inserted, level must be <= budget
      cstName = ddPrefix + "_5";
      model.AddLinearConstraint(
          resultAfterBootstrapVar <=
              levelBudget + IF_THEN_AUX * insertBootstrapDecision,
          cstName);
    }
  });

  // Dump the model
  LLVM_DEBUG({
    std::stringstream ss;
    ss << model;
    llvm::dbgs() << ss.str();
  });

  const absl::StatusOr<math_opt::SolveResult> status =
      math_opt::Solve(model, math_opt::SolverType::kGscip);

  if (!status.ok()) {
    std::stringstream ss;
    ss << "Error solving the problem: " << status.status() << "\n";
    llvm::errs() << ss.str();
    return failure();
  }

  const math_opt::SolveResult& result = status.value();

  switch (result.termination.reason) {
    case math_opt::TerminationReason::kOptimal:
    case math_opt::TerminationReason::kFeasible:
      LLVM_DEBUG({
        llvm::dbgs() << "Problem solved in "
                     << result.solve_time() / absl::Milliseconds(1)
                     << " milliseconds.\n"
                     << "Solution:\n";
        llvm::dbgs() << "Objective value = " << result.objective_value()
                     << "\n";
        for (const auto& [var, value] : result.variable_values()) {
          llvm::dbgs() << var.name() << " = " << value << "\n";
        }
      });
      break;
    default:
      llvm::errs() << "The problem does not have a feasible solution. "
                      "Termination status code: "
                   << (int)result.termination.reason
                   << " (see: "
                      "https://github.com/google/or-tools/blob/"
                      "ed94162b910fa58896db99191378d3b71a5313af/ortools/"
                      "math_opt/cpp/solve_result.h#L124)"
                   << "\n";
      return failure();
  }

  auto varMap = result.variable_values();

  // Debug output: Print solution summary
  llvm::errs() << "\n=== Bootstrap ILP Solution (levelBudget=" << levelBudget
               << ") ===\n";
  llvm::errs() << "Objective value (number of bootstraps): "
               << result.objective_value() << "\n";

  int bootstrapCount = 0;
  int totalDecisions = 0;
  for (auto item : decisionVariables) {
    totalDecisions++;
    bool shouldBootstrap = varMap[item.second] > 0.5;
    if (shouldBootstrap) {
      bootstrapCount++;
      llvm::errs() << "  INSERT BOOTSTRAP after: " << item.first->getName()
                   << " (op ptr: " << (void*)item.first << ")\n";
    }
    solution.insert(std::make_pair(item.first, shouldBootstrap));
  }
  llvm::errs() << "Total decision variables: " << totalDecisions << "\n";
  llvm::errs() << "Total bootstraps to insert: " << bootstrapCount << "\n";

  // Print some level variables for debugging
  llvm::errs() << "\nSample level variables:\n";
  int sampleCount = 0;
  for (auto item : levelVars) {
    if (sampleCount++ >= 10) break;  // Limit output
    double level = varMap[item.second];
    llvm::errs() << "  Level_" << (sampleCount - 1) << " = " << level << "\n";
  }

  for (auto item : beforeBootstrapVars) {
    solutionLevelBeforeBootstrap.insert(
        std::make_pair(item.first, (int)varMap[item.second]));
  }
  llvm::errs() << "==========================\n\n";

  return success();
}

}  // namespace heir
}  // namespace mlir
