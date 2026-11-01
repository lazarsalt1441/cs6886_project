// RUN: iree-opt --pass-pipeline='builtin.module(func.func(iree-codegen-generic-vectorization{enable-vector-masking=true}))' --split-input-file %s | FileCheck %s

// The optional trailing unit axis stores the index component, not a batch.
// CHECK-DAG: #[[BATCH:.*]] = affine_map<(d0, d1) -> (d0)>
// CHECK-LABEL: @explicit_index_component
// CHECK: vector.transfer_read {{.*}} {permutation_map = #[[BATCH]]}
// CHECK-SAME: tensor<16x1xi64>, vector<16xi64>
func.func @explicit_index_component(%src: tensor<257x4xf32>, %idx: tensor<16x1xi64>) -> tensor<16x4xf32> {
  %empty = tensor.empty() : tensor<16x4xf32>
  %r = iree_linalg_ext.gather dimension_map = [0] ins(%src, %idx : tensor<257x4xf32>, tensor<16x1xi64>) outs(%empty : tensor<16x4xf32>) -> tensor<16x4xf32>
  return %r : tensor<16x4xf32>
}

// -----

// CHECK-DAG: #[[BATCH2:.*]] = affine_map<(d0, d1, d2) -> (d0, d1)>
// CHECK-LABEL: @two_batch_axes
// CHECK: vector.transfer_read {{.*}} {permutation_map = #[[BATCH2]]}
// CHECK-SAME: tensor<2x8x1xi32>, vector<2x8xi32>
func.func @two_batch_axes(%src: tensor<257x4xf32>, %idx: tensor<2x8x1xi32>) -> tensor<2x8x4xf32> {
  %empty = tensor.empty() : tensor<2x8x4xf32>
  %r = iree_linalg_ext.gather dimension_map = [0] ins(%src, %idx : tensor<257x4xf32>, tensor<2x8x1xi32>) outs(%empty : tensor<2x8x4xf32>) -> tensor<2x8x4xf32>
  return %r : tensor<2x8x4xf32>
}
