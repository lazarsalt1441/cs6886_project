module {
func.func @main(%q: tensor<8x1x16xf32>, %k: tensor<2x32x16xf32>) -> tensor<8x1x32xf32> {
%ke = tensor.expand_shape %k [[0,1],[2],[3]] output_shape [2,1,32,16] : tensor<2x32x16xf32> into tensor<2x1x32x16xf32>
%empty = tensor.empty() : tensor<2x4x32x16xf32>
%kb = linalg.generic {indexing_maps = [affine_map<(h,g,n,d)->(h,0,n,d)>, affine_map<(h,g,n,d)->(h,g,n,d)>], iterator_types = ["parallel","parallel","parallel","parallel"]} ins(%ke : tensor<2x1x32x16xf32>) outs(%empty : tensor<2x4x32x16xf32>) {
^bb0(%x:f32,%o:f32):
linalg.yield %x : f32
} -> tensor<2x4x32x16xf32>
%kc = tensor.collapse_shape %kb [[0,1],[2],[3]] : tensor<2x4x32x16xf32> into tensor<8x32x16xf32>
%zero = arith.constant dense<0.0> : tensor<8x1x32xf32>
%r = linalg.batch_matmul indexing_maps = [affine_map<(b,m,n,k)->(b,m,k)>,affine_map<(b,m,n,k)->(b,n,k)>,affine_map<(b,m,n,k)->(b,m,n)>] ins(%q,%kc : tensor<8x1x16xf32>, tensor<8x32x16xf32>) outs(%zero : tensor<8x1x32xf32>) -> tensor<8x1x32xf32>

return %r : tensor<8x1x32xf32>
}
}
