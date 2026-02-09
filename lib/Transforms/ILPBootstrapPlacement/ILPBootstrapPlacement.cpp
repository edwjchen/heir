#include "lib/Transforms/ILPBootstrapPlacement/ILPBootstrapPlacement.h"

#include "lib/Analysis/ILPBootstrapPlacementAnalysis/ILPBootstrapPlacementAnalysis.h"
#include "lib/Analysis/SecretnessAnalysis/SecretnessAnalysis.h"
#include "lib/Dialect/Mgmt/IR/MgmtDialect.h"
#include "lib/Dialect/Mgmt/IR/MgmtOps.h"
#include "lib/Dialect/Secret/IR/SecretOps.h"
#include "llvm/include/llvm/Support/Debug.h"               // from @llvm-project
#include "mlir/include/mlir/Analysis/DataFlow/Utils.h"     // from @llvm-project
#include "mlir/include/mlir/Analysis/DataFlowFramework.h"  // from @llvm-project
#include "mlir/include/mlir/IR/Builders.h"                 // from @llvm-project
#include "mlir/include/mlir/IR/MLIRContext.h"              // from @llvm-project
#include "mlir/include/mlir/IR/Value.h"                    // from @llvm-project
#include "mlir/include/mlir/IR/Visitors.h"                  // from @llvm-project
#include "mlir/include/mlir/Support/LLVM.h"                // from @llvm-project

namespace mlir {
namespace heir {

#define DEBUG_TYPE "ILPBootstrapPlacement"

#define GEN_PASS_DEF_ILPBOOTSTRAPPLACEMENT
#include "lib/Transforms/ILPBootstrapPlacement/ILPBootstrapPlacement.h.inc"

struct ILPBootstrapPlacement
    : impl::ILPBootstrapPlacementBase<ILPBootstrapPlacement> {
  using ILPBootstrapPlacementBase::ILPBootstrapPlacementBase;

  void processSecretGenericOp(secret::GenericOp genericOp,
                              DataFlowSolver* solver) {
    // Remove all bootstrap ops. This makes the IR invalid, because the level
    // states are incorrect. However, the correctness of the ILP ensures the
    // level states are made correct at the end.
    genericOp->walk([&](mgmt::BootstrapOp op) {
      op.getResult().replaceAllUsesWith(op.getOperand());
      op.erase();
    });

    ILPBootstrapPlacementAnalysis analysis(
        genericOp, solver, useLocBasedVariableNames);
    if (failed(analysis.solve())) {
      genericOp->emitError("Failed to solve the bootstrap placement optimization problem");
      return signalPassFailure();
    }

    OpBuilder b(&getContext());

    genericOp->walk([&](Operation* op) {
      if (!analysis.shouldInsertBootstrap(op)) return;

      LLVM_DEBUG(llvm::dbgs()
                 << "Inserting bootstrap after: " << op->getName() << "\n");

      b.setInsertionPointAfter(op);
      for (Value result : op->getResults()) {
        // Only insert bootstrap for secret values
        if (!isa<secret::SecretType>(result.getType())) continue;
        
        auto bootstrapOp = mgmt::BootstrapOp::create(
            b, op->getLoc(), result);
        result.replaceAllUsesExcept(bootstrapOp.getResult(), {bootstrapOp});
      }
    });
  }

  void runOnOperation() override {
    Operation* module = getOperation();

    DataFlowSolver solver;
    dataflow::loadBaselineAnalyses(solver);
    solver.load<SecretnessAnalysis>();

    if (failed(solver.initializeAndRun(getOperation()))) {
      getOperation()->emitOpError() << "Failed to run the analysis.\n";
      signalPassFailure();
      return;
    }

    module->walk(
        [&](secret::GenericOp op) { processSecretGenericOp(op, &solver); });
  }
};

}  // namespace heir
}  // namespace mlir
