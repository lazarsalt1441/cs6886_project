// Independent of Torch export and cache update optimization.
// With zero Q/K, IDs [0,2,...,254], and V[p,h,d] = 2*p+h,
// every output feature must be [254,255] across the two heads.
#q = affine_map<(h,m,d,k,n)->(h,m,k)>
#k = affine_map<(h,m,d,k,n)->(h,n,k)>
#v = affine_map<(h,m,d,k,n)->(h,n,d)>
#s = affine_map<(h,m,d,k,n)->()>
#o = affine_map<(h,m,d,k,n)->(h,m,d)>
module {
 func.func @main(%q: tensor<2x1x32xf32>, %pk: tensor<257x2x32xf32>, %pv: tensor<257x2x32xf32>, %ids: tensor<128x1xi64>) -> tensor<2x1x32xf32> {
  %e = tensor.empty() : tensor<128x2x32xf32>
  %kg = iree_linalg_ext.gather dimension_map=[0] ins(%pk,%ids: tensor<257x2x32xf32>,tensor<128x1xi64>) outs(%e: tensor<128x2x32xf32>) -> tensor<128x2x32xf32>
  %vg = iree_linalg_ext.gather dimension_map=[0] ins(%pv,%ids: tensor<257x2x32xf32>,tensor<128x1xi64>) outs(%e: tensor<128x2x32xf32>) -> tensor<128x2x32xf32>
  %te = tensor.empty() : tensor<2x128x32xf32>
  %kt = linalg.transpose ins(%kg: tensor<128x2x32xf32>) outs(%te: tensor<2x128x32xf32>) permutation=[1,0,2]
  %vt = linalg.transpose ins(%vg: tensor<128x2x32xf32>) outs(%te: tensor<2x128x32xf32>) permutation=[1,0,2]
  %scale = arith.constant 0.1767766952966369 : f32
  %oe = tensor.empty() : tensor<2x1x32xf32>
  %out = iree_linalg_ext.attention <indexing_maps=[#q,#k,#v,#s,#o]>
    ins(%q,%kt,%vt,%scale: tensor<2x1x32xf32>,tensor<2x128x32xf32>,tensor<2x128x32xf32>,f32)
    outs(%oe: tensor<2x1x32xf32>) {
      ^bb0(%score: f32):
      iree_linalg_ext.yield %score: f32
    } -> tensor<2x1x32xf32>
  return %out: tensor<2x1x32xf32>
 }
}
