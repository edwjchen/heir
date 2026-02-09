#ifndef LIB_TRANSFORMS_ILP_BOOTSTRAP_PLACEMENT_TD_
#define LIB_TRANSFORMS_ILP_BOOTSTRAP_PLACEMENT_TD_

include "mlir/Pass/PassBase.td"

def ILPBootstrapPlacement : Pass <"ilp-bootstrap-placement"> {
    let summary = "Optimize placement of bootstrap ops using ILP";
    let description = [{
        This pass uses an integer linear program to determine the optimal
        placement of bootstrap operations in the IR. It solves an ILP for each
        `func` op in the IR.

        The pass walks the input program, parses it into ILP constraints,
        solves where to place bootstrap operations, and modifies the input
        program to include bootstrap placement.

        The assumptions of this pass include:

        - All bootstrap operations are removed from the input IR before solving
        - The ILP model determines optimal bootstrap placement based on level
          constraints and costs
        - Bootstrap operations reset the multiplicative depth/level of ciphertexts

        The ILP formulation will be added in a future update.
    }];

    let dependentDialects = ["mlir::heir::mgmt::MgmtDialect"];

  let options = [
    // When using this option, one should generally write the IR to disk and
    // run `--ilp-bootstrap-placement` as the first or only pass, otherwise the
    // ILP model construction may fail due to having non-unique variable names
    Option<"useLocBasedVariableNames",
           "use-loc-based-variable-names",
           "bool",
           /*default=*/"false",
           "When true, the ILP uses op source locations in variable names, "
           "which can help debug ILP model bugs.">,
  ];
}

#endif  // LIB_TRANSFORMS_ILP_BOOTSTRAP_PLACEMENT_TD_
