module {
func.func @main(%q: tensor<1x1x128xf32>, %k: tensor<1x32x128xf32>, %v: tensor<1x32x128xf32>) -> tensor<1x1x128xf32> {
%scale = arith.constant 8.838834764832e-02 : f32
%e = tensor.empty() : tensor<1x1x128xf32>
%r = iree_linalg_ext.attention <indexing_maps = [affine_map<(b,m,k,n,v)->(b,m,k)>,affine_map<(b,m,k,n,v)->(b,n,k)>,affine_map<(b,m,k,n,v)->(b,n,v)>,affine_map<(b,m,k,n,v)->()>,affine_map<(b,m,k,n,v)->(b,m,v)>]> ins(%q,%k,%v,%scale : tensor<1x1x128xf32>, tensor<1x32x128xf32>, tensor<1x32x128xf32>, f32) outs(%e : tensor<1x1x128xf32>) {
^bb0(%score:f32):
iree_linalg_ext.yield %score : f32
} -> tensor<1x1x128xf32>
return %r : tensor<1x1x128xf32>
}
}
