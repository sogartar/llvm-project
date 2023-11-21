// RUN: mlir-opt -test-mesh-simplifications %s | FileCheck %s

mesh.cluster @mesh0(rank = 1, dim_sizes = [4])

// CHECK-LABEL: func.func @all_reduce_arith_addf
func.func @all_reduce_arith_addf(
    // CHECK-SAME: %[[ARG0:[A-Za-z0-9_]*]]: tensor<5xf32>
    %arg0: tensor<5xf32>,
    // CHECK-SAME: %[[ARG1:[A-Za-z0-9_]*]]: tensor<5xf32>
    %arg1: tensor<5xf32>) -> tensor<5xf32> {
  %0 = mesh.all_reduce %arg0 on @mesh0 mesh_axes = [0]
    : tensor<5xf32> -> tensor<5xf32>
  %1 = mesh.all_reduce %arg1 on @mesh0 mesh_axes = [0]
    : tensor<5xf32> -> tensor<5xf32>
  // CHECK: %[[ADD_RES:[A-Za-z0-9_]*]] = arith.addf %[[ARG0]], %[[ARG1]]
  %2 = arith.addf %0, %1 : tensor<5xf32>
  // CHECK: %[[ALL_REDUCE_RES:[A-Za-z0-9_]*]] = mesh.all_reduce %[[ADD_RES]]
  // CHECK: return %[[ALL_REDUCE_RES]]
  return %2 : tensor<5xf32>
}
