module {
func.func @main(%base: tensor<4096xf32>, %a: tensor<16xf32>, %b: tensor<16xf32>, %cond: tensor<i1>) -> tensor<4096xf32> {
%u = tensor.insert_slice %a into %base[128] [16] [1] : tensor<16xf32> into tensor<4096xf32>
%v = tensor.insert_slice %b into %base[128] [16] [1] : tensor<16xf32> into tensor<4096xf32>
%e = tensor.empty() : tensor<4096xf32>
%r = linalg.generic {indexing_maps=[affine_map<(i)->()>,affine_map<(i)->(i)>,affine_map<(i)->(i)>,affine_map<(i)->(i)>],iterator_types=["parallel"]} ins(%cond,%u,%v : tensor<i1>,tensor<4096xf32>,tensor<4096xf32>) outs(%e : tensor<4096xf32>) {
^bb0(%gc:i1,%ga:f32,%gb:f32,%go:f32):
%selected = arith.select %gc,%ga,%gb : f32
linalg.yield %selected : f32
} -> tensor<4096xf32>
return %r : tensor<4096xf32>
}
}
