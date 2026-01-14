#include "lib/Transforms/SecretInsertMgmt/Pipeline.h"

#include <utility>

#include "lib/Analysis/LevelAnalysis/LevelAnalysis.h"
#include "lib/Analysis/MulDepthAnalysis/MulDepthAnalysis.h"
#include "lib/Analysis/OptimizeBootstrapILPAnalysis/OptimizeBootstrapILPAnalysis.h"
#include "lib/Analysis/SecretnessAnalysis/SecretnessAnalysis.h"
#include "lib/Dialect/Mgmt/IR/MgmtOps.h"
#include "lib/Dialect/Secret/IR/SecretOps.h"
#include "lib/Transforms/SecretInsertMgmt/SecretInsertMgmtPatterns.h"
#include "llvm/include/llvm/Support/Debug.h"               // from @llvm-project
#include "mlir/include/mlir/Analysis/DataFlow/Utils.h"     // from @llvm-project
#include "mlir/include/mlir/Analysis/DataFlowFramework.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/Arith/IR/Arith.h"      // from @llvm-project
#include "mlir/include/mlir/Dialect/Tensor/IR/Tensor.h"    // from @llvm-project
#include "mlir/include/mlir/IR/Builders.h"                 // from @llvm-project
#include "mlir/include/mlir/IR/PatternMatch.h"             // from @llvm-project
#include "mlir/include/mlir/IR/Visitors.h"                 // from @llvm-project
#include "mlir/include/mlir/Pass/PassManager.h"            // from @llvm-project
#include "mlir/include/mlir/Support/LLVM.h"                // from @llvm-project
#include "mlir/include/mlir/Transforms/WalkPatternRewriteDriver.h"  // from @llvm-project

#define DEBUG_TYPE "secret-insert-mgmt"

namespace mlir {
namespace heir {

LogicalResult runInsertMgmtPipeline(Operation* top,
                                    const InsertMgmtPipelineOptions& options) {
  DataFlowSolver solver;
  dataflow::loadBaselineAnalyses(solver);
  solver.load<SecretnessAnalysis>();
  solver.load<LevelAnalysis>();
  solver.load<MulDepthAnalysis>();

  if (failed(solver.initializeAndRun(top))) {
    top->emitOpError() << "Failed to run the analysis.\n";
    return failure();
  }

  insertMgmtInitForPlaintexts(top, solver, options.includeFloats);
  insertModReduceBeforeOrAfterMult(top, solver, options.modReduceAfterMul,
                                   options.modReduceBeforeMulIncludeFirstMul,
                                   options.includeFloats);
  rerunDataflow(solver, top);

  // this must be run after ModReduceAfterMult
  insertRelinearizeAfterMult(top, solver, options.includeFloats);
  rerunDataflow(solver, top);

  // insert BootstrapOp using ILP optimizations
  // TODO: add more details here.
  if (options.bootstrapILPLevelBudget.has_value()) {
    insertBootstrapILP(top, solver, options.bootstrapILPLevelBudget.value());
    rerunDataflow(solver, top);
  }

  // insert BootstrapOp after mgmt::ModReduceOp
  // This must be run before level mismatch
  // NOTE: actually bootstrap before mod reduce is better
  // as after modreduce to level `0` there still might be add/sub
  // and these op done there could be minimal cost.
  // However, this greedy strategy is temporary so not too much
  // optimization now
  else if (options.bootstrapWaterline.has_value()) {
    insertBootstrapWaterLine(top, solver, options.bootstrapWaterline.value());
    rerunDataflow(solver, top);
  }

  int idCounter = 0;  // for making adjust_scale op different to avoid cse
  handleCrossLevelOps(top, solver, &idCounter, options.includeFloats);
  rerunDataflow(solver, top);
  handleCrossMulDepthOps(top, solver, &idCounter, options.includeFloats);
  return success();
}

void rerunDataflow(DataFlowSolver& solver, Operation* top) {
  LLVM_DEBUG(llvm::dbgs() << "Re-running dataflow\n");
  solver.eraseAllStates();
  (void)solver.initializeAndRun(top);
}

void insertMgmtInitForPlaintexts(Operation* top, DataFlowSolver& solver,
                                 bool includeFloats) {
  MLIRContext* ctx = top->getContext();
  LLVM_DEBUG(llvm::dbgs() << "Insert Mgmt Init for Plaintext Operands\n");

  RewritePatternSet patterns(ctx);
  patterns.add<UseInitOpForPlaintextOperand<arith::AddIOp>,
               UseInitOpForPlaintextOperand<arith::SubIOp>,
               UseInitOpForPlaintextOperand<arith::MulIOp>,
               UseInitOpForPlaintextOperand<tensor::ExtractSliceOp>,
               UseInitOpForPlaintextOperand<tensor::InsertSliceOp>>(ctx, top,
                                                                    &solver);

  if (includeFloats) {
    patterns.add<UseInitOpForPlaintextOperand<arith::AddFOp>,
                 UseInitOpForPlaintextOperand<arith::SubFOp>,
                 UseInitOpForPlaintextOperand<arith::MulFOp>>(ctx, top,
                                                              &solver);
  }

  (void)walkAndApplyPatterns(top, std::move(patterns));
}

void insertModReduceBeforeOrAfterMult(Operation* top, DataFlowSolver& solver,
                                      bool afterMul,
                                      bool beforeMulIncludeFirstMul,
                                      bool includeFloats) {
  MLIRContext* ctx = top->getContext();
  LLVM_DEBUG({
    auto when = "before mul";
    if (afterMul) when = "after mul";
    if (beforeMulIncludeFirstMul) when = "before mul + before first mul";
    llvm::dbgs() << "Insert ModReduce " << when << "\n";
  });

  RewritePatternSet patterns(ctx);
  if (afterMul) {
    patterns.add<ModReduceAfterMult<arith::MulIOp>>(ctx, top, &solver);
    if (includeFloats)
      patterns.add<ModReduceAfterMult<arith::MulFOp>>(ctx, top, &solver);
  } else {
    patterns.add<ModReduceBefore<arith::MulIOp>>(ctx, beforeMulIncludeFirstMul,
                                                 top, &solver);
    if (includeFloats)
      patterns.add<ModReduceBefore<arith::MulFOp>>(
          ctx, beforeMulIncludeFirstMul, top, &solver);
    // includeFirstMul = false here
    // as before yield we only want mulResult to be mod reduced
    patterns.add<ModReduceBefore<secret::YieldOp>>(
        ctx, /*includeFirstMul*/ false, top, &solver);
  }
  (void)walkAndApplyPatterns(top, std::move(patterns));
}

void insertRelinearizeAfterMult(Operation* top, DataFlowSolver& solver,
                                bool includeFloats) {
  MLIRContext* ctx = top->getContext();
  LLVM_DEBUG(llvm::dbgs() << "Insert Relinearize After Mult\n");
  RewritePatternSet patterns(ctx);
  patterns.add<MultRelinearize<arith::MulIOp>>(ctx, top, &solver);
  if (includeFloats)
    patterns.add<MultRelinearize<arith::MulFOp>>(ctx, top, &solver);
  (void)walkAndApplyPatterns(top, std::move(patterns));
}

void handleCrossLevelOps(Operation* top, DataFlowSolver& solver, int* idCounter,
                         bool includeFloats) {
  MLIRContext* ctx = top->getContext();
  LLVM_DEBUG(llvm::dbgs() << "Handle Cross Level Ops\n");
  RewritePatternSet patterns(ctx);
  patterns.add<MatchCrossLevel<arith::AddIOp>, MatchCrossLevel<arith::SubIOp>,
               MatchCrossLevel<arith::MulIOp>>(ctx, idCounter, top, &solver);
  if (includeFloats)
    patterns.add<MatchCrossLevel<arith::AddFOp>, MatchCrossLevel<arith::SubFOp>,
                 MatchCrossLevel<arith::MulFOp>>(ctx, idCounter, top, &solver);
  (void)walkAndApplyPatterns(top, std::move(patterns));
}

// this only happen for before-mul but not include-first-mul case
// at the first level, a Value can be both mulResult or not mulResult
// we should match their scale by adding one adjust scale op
void handleCrossMulDepthOps(Operation* top, DataFlowSolver& solver,
                            int* idCounter, bool includeFloats) {
  MLIRContext* ctx = top->getContext();
  LLVM_DEBUG(llvm::dbgs() << "Handle Cross MulDepth Ops\n");
  RewritePatternSet patterns(ctx);
  patterns
      .add<MatchCrossMulDepth<arith::AddIOp>, MatchCrossMulDepth<arith::SubIOp>,
           MatchCrossMulDepth<arith::MulIOp>>(ctx, idCounter, top, &solver);
  if (includeFloats)
    patterns.add<MatchCrossMulDepth<arith::AddFOp>,
                 MatchCrossMulDepth<arith::SubFOp>,
                 MatchCrossMulDepth<arith::MulFOp>>(ctx, idCounter, top,
                                                    &solver);
  (void)walkAndApplyPatterns(top, std::move(patterns));
}

void insertBootstrapILP(Operation* top, DataFlowSolver& solver,
                        int levelBudget) {
  MLIRContext* ctx = top->getContext();
  LLVM_DEBUG(llvm::dbgs() << "Insert Bootstrap using ILP optimization\n");

  // Process each secret::GenericOp similar to OptimizeRelinearization
  top->walk([&](secret::GenericOp genericOp) {
    // Remove all existing bootstrap ops. This makes the IR invalid, because the
    // level information is incorrect. However, the correctness of the ILP
    // ensures the level information is made correct at the end.
    genericOp->walk([&](mgmt::BootstrapOp op) {
      op.getResult().replaceAllUsesWith(op.getOperand());
      op.erase();
    });

    // Run the bootstrap ILP analysis
    OptimizeBootstrapILPAnalysis analysis(genericOp, &solver, levelBudget);
    if (failed(analysis.solve())) {
      genericOp->emitError(
          "Failed to solve the bootstrap optimization problem");
      return;
    }

    OpBuilder builder(ctx);

    genericOp->walk([&](Operation* op) {
      bool shouldInsert = analysis.shouldInsertBootstrap(op);
      // Only print for operations that might have bootstrap decisions
      if (isa<arith::MulFOp>(op) || shouldInsert) {
        llvm::errs() << "Checking operation: " << op->getName()
                     << " (op ptr: " << (void*)op
                     << ") -> shouldInsertBootstrap = " << shouldInsert << "\n";
      }
      if (!shouldInsert) return;

      llvm::errs() << "INSERTING bootstrap after: " << op->getName() << "\n";
      LLVM_DEBUG(llvm::dbgs()
                 << "Inserting bootstrap after: " << op->getName() << "\n");

      builder.setInsertionPointAfter(op);
      for (Value result : op->getResults()) {
        if (!isSecret(result, &solver)) {
          continue;
        }
        auto bootstrapOp = mgmt::BootstrapOp::create(builder, op->getLoc(),
                                                     result.getType(), result);
        result.replaceAllUsesExcept(bootstrapOp, {bootstrapOp});

        // Update solver state
        auto* secretnessLattice =
            solver.getOrCreateState<SecretnessLattice>(bootstrapOp);
        secretnessLattice->getValue().setSecretness(true);
      }
    });
  });
}

void insertBootstrapWaterLine(Operation* top, DataFlowSolver& solver,
                              int bootstrapWaterline) {
  MLIRContext* ctx = top->getContext();
  LLVM_DEBUG(llvm::dbgs() << "Insert Bootstrap at Water Line\n");

  RewritePatternSet patterns(ctx);
  patterns.add<BootstrapWaterLine<mgmt::ModReduceOp>>(ctx, top, &solver,
                                                      bootstrapWaterline);
  (void)walkAndApplyPatterns(top, std::move(patterns));
}
}  // namespace heir
}  // namespace mlir
