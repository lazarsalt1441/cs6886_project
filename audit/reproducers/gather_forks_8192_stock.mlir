module {
func.func @main(%base: tensor<8192x16xf32>, %a: tensor<4x16xf32>, %b: tensor<4x16xf32>, %idx: tensor<8xi64>) -> tensor<16x16xf32> {
%ua = tensor.insert_slice %a into %base[128,0] [4,16] [1,1] : tensor<4x16xf32> into tensor<8192x16xf32>
%ub = tensor.insert_slice %b into %base[256,0] [4,16] [1,1] : tensor<4x16xf32> into tensor<8192x16xf32>
%gae = tensor.empty() : tensor<8x16xf32>
%ga = linalg.generic {indexing_maps=[affine_map<(i,j)->(i)>,affine_map<(i,j)->(i,j)>],iterator_types=["parallel","parallel"]} ins(%idx : tensor<8xi64>) outs(%gae : tensor<8x16xf32>) {
^bb0(%raw:i64,%unused:f32):
%row = arith.index_cast %raw : i64 to index
%col = linalg.index 1 : index
%x = tensor.extract %ua[%row,%col] : tensor<8192x16xf32>
linalg.yield %x : f32
} -> tensor<8x16xf32>
%gbe = tensor.empty() : tensor<8x16xf32>
%gb = linalg.generic {indexing_maps=[affine_map<(i,j)->(i)>,affine_map<(i,j)->(i,j)>],iterator_types=["parallel","parallel"]} ins(%idx : tensor<8xi64>) outs(%gbe : tensor<8x16xf32>) {
^bb0(%raw:i64,%unused:f32):
%row = arith.index_cast %raw : i64 to index
%col = linalg.index 1 : index
%x = tensor.extract %ub[%row,%col] : tensor<8192x16xf32>
linalg.yield %x : f32
} -> tensor<8x16xf32>
%r = tensor.concat dim(0) %ga,%gb : (tensor<8x16xf32>,tensor<8x16xf32>) -> tensor<16x16xf32>
return %r : tensor<16x16xf32>
}
}
