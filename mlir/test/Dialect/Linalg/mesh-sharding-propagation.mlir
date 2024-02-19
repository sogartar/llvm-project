// RUN: mlir-opt \
// RUN:  --mesh-sharding-propagation \
// RUN:  --split-input-file \
// RUN:  %s | FileCheck %s

#map_identity_1d = affine_map<(d0) -> (d0)>

mesh.mesh @mesh_1d(shape = 2)

// CHECK-LABEL: func @elementwise_static_1d_mesh_static_1d_tensor
func.func @elementwise_static_1d_mesh_static_1d_tensor(
  // CHECK-SAME: %[[IN1:[A-Za-z0-9_]+]]: tensor<1xi8>,
  %in1: tensor<2xi8>,
  // CHECK-SAME: %[[IN2:[A-Za-z0-9_]+]]: tensor<1xi8>,
  %in2: tensor<2xi8>,
  // CHECK-SAME: %[[DPS_OUT:[A-Za-z0-9_]+]]: tensor<1xi8>
  %dps_out: tensor<2xi8>
// CHECK-SAME: -> tensor<1xi8> {
) -> tensor<2xi8> {
  %in1_shared1 = mesh.shard %in1 to <@mesh_1d, [[0]]> : tensor<2xi8>
  %in1_shared2 = mesh.shard %in1_shared1 to <@mesh_1d, [[0]]> annotate_for_users: tensor<2xi8>
  %res = linalg.generic {
      indexing_maps = [#map_identity_1d, #map_identity_1d, #map_identity_1d],
      iterator_types = ["parallel"]
    } ins(%in1_shared2, %in2 : tensor<2xi8>, tensor<2xi8>)
      outs(%dps_out : tensor<2xi8>) {
    ^bb0(%in1_scalar: i8, %in2_scalar: i8, %out: i8):
      %res_scalar = arith.muli %in1_scalar, %in2_scalar : i8
      linalg.yield %res_scalar : i8
    } -> tensor<2xi8>
  // CHECK: return %[[RES]] : tensor<1xi8>
  return %res : tensor<2xi8>
}
