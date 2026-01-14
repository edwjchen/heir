#ifndef LIB_ANALYSIS_OPTIMIZE_BOOTSTRAP_ILP_ANALYSIS_H
#define LIB_ANALYSIS_OPTIMIZE_BOOTSTRAP_ILP_ANALYSIS_H

#include "llvm/include/llvm/ADT/DenseMap.h"                // from @llvm-project
#include "mlir/include/mlir/Analysis/DataFlowFramework.h"  // from @llvm-project
#include "mlir/include/mlir/IR/Operation.h"                // from @llvm-project
#include "mlir/include/mlir/IR/Value.h"                    // from @llvm-project
#include "mlir/include/mlir/Support/LLVM.h"                // from @llvm-project

namespace mlir {
namespace heir {
class OptimizeBootstrapILPAnalysis {
 public:
  OptimizeBootstrapILPAnalysis(Operation* op, DataFlowSolver* solver,
                               int levelBudget)
      : opToRunOn(op), solver(solver), levelBudget(levelBudget) {}
  ~OptimizeBootstrapILPAnalysis() = default;

  LogicalResult solve();

  // Return true if a bootstrap op should be inserted after the given
  // operation, according to the solution to the optimization problem.
  bool shouldInsertBootstrap(Operation* op) const {
    return solution.lookup(op);
  }

  // Return the level at the given SSA value before bootstrap is applied,
  // as determined by the solution to the optimization problem.
  // When the input value is the result of an op, and the model solution
  // suggests a bootstrap should be inserted after that op, this function
  // returns the pre-bootstrap level.
  //
  // After bootstrap, the level is reset to 0.
  int levelBeforeBootstrap(Value value) const {
    return solutionLevelBeforeBootstrap.lookup(value);
  }

 private:
  Operation* opToRunOn;
  DataFlowSolver* solver;
  int levelBudget;
  llvm::DenseMap<Operation*, bool> solution;
  llvm::DenseMap<Value, int> solutionLevelBeforeBootstrap;
};
}  // namespace heir
}  // namespace mlir

#endif  // LIB_ANALYSIS_OPTIMIZE_BOOTSTRAP_ILP_ANALYSIS_H
