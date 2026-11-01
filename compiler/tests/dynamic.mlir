#src = affine_map<(b,h,r,n,d)->(b,h,n,d)>
#dst = affine_map<(b,h,r,n,d)->(b,h,r,n,d)>
module {
 func.func @dynamic(%q: tensor<8x1x?xf32>, %kv: tensor<1x2x?x32xf32>, %init: tensor<8x1x32xf32>) -> tensor<8x1x32xf32> {
  %c2 = arith.constant 2 : index
  %n = tensor.dim %kv, %c2 : tensor<1x2x?x32xf32>
  %empty = tensor.empty(%n) : tensor<1x2x4x?x32xf32>
  %rep = linalg.generic {indexing_maps=[#src,#dst],iterator_types=["parallel","parallel","parallel","parallel","parallel"]} ins(%kv:tensor<1x2x?x32xf32>) outs(%empty:tensor<1x2x4x?x32xf32>) {
  ^bb0(%x:f32,%y:f32): linalg.yield %x : f32
  } -> tensor<1x2x4x?x32xf32>
  %rhs = tensor.collapse_shape %rep [[0,1,2],[3],[4]] : tensor<1x2x4x?x32xf32> into tensor<8x?x32xf32>
  %out = linalg.batch_matmul ins(%q,%rhs:tensor<8x1x?xf32>,tensor<8x?x32xf32>) outs(%init:tensor<8x1x32xf32>) -> tensor<8x1x32xf32>
  return %out : tensor<8x1x32xf32>
 }
}
