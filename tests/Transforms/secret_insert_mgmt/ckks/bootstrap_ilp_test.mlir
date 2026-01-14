// RUN: heir-opt --mlir-to-secret-arithmetic --secret-insert-mgmt-ckks %s 2>&1 | FileCheck %s
//
// This test exercises the insertBootstrapILP function with a long chain of
// sequential multiplications (20 multiplications) to test the ILP optimization.

func.func @bootstrap_ilp_test(
    %x : f16 {secret.secret}
  ) -> f16 {
    %0 = arith.mulf %x, %x : f16
    %1 = arith.mulf %0, %0 : f16
    %2 = arith.mulf %1, %1 : f16
    %3 = arith.mulf %2, %2 : f16
    %4 = arith.mulf %3, %3 : f16
    %5 = arith.mulf %4, %4 : f16
    %6 = arith.mulf %5, %5 : f16
    %7 = arith.mulf %6, %6 : f16
    %8 = arith.mulf %7, %7 : f16
    %9 = arith.mulf %8, %8 : f16
    %10 = arith.mulf %9, %9 : f16
    %11 = arith.mulf %10, %10 : f16
    %12 = arith.mulf %11, %11 : f16
    %13 = arith.mulf %12, %12 : f16
    %14 = arith.mulf %13, %13 : f16
    %15 = arith.mulf %14, %14 : f16
    %16 = arith.mulf %15, %15 : f16
    %17 = arith.mulf %16, %16 : f16
    %18 = arith.mulf %17, %17 : f16
    %19 = arith.mulf %18, %18 : f16
    %20 = arith.mulf %19, %19 : f16
    return %20 : f16
}
