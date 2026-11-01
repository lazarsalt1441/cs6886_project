module {
func.func @main(%q: tensor<1x128xf32>, %l: tensor<512x32xf32>, %wk: tensor<32x128xf32>, %wv: tensor<32x128xf32>) -> tensor<1x128xf32> {
%keyz = arith.constant dense<0.0> : tensor<512x128xf32>
%key = linalg.matmul ins(%l,%wk : tensor<512x32xf32>, tensor<32x128xf32>) outs(%keyz : tensor<512x128xf32>) -> tensor<512x128xf32>
%scorez = arith.constant dense<0.0> : tensor<1x512xf32>
%score = linalg.matmul indexing_maps = [affine_map<(m,n,k)->(m,k)>,affine_map<(m,n,k)->(n,k)>,affine_map<(m,n,k)->(m,n)>] ins(%q,%key : tensor<1x128xf32>, tensor<512x128xf32>) outs(%scorez : tensor<1x512xf32>) -> tensor<1x512xf32>
%scale = arith.constant dense<8.838834764832e-02> : tensor<1x512xf32>
%scaled = arith.mulf %score, %scale : tensor<1x512xf32>
%neg = arith.constant dense<0xFF800000> : tensor<1xf32>
%mx = linalg.generic {indexing_maps = [affine_map<(m,n)->(m,n)>,affine_map<(m,n)->(m)>], iterator_types = ["parallel","reduction"]} ins(%scaled : tensor<1x512xf32>) outs(%neg : tensor<1xf32>) {
^bb0(%x:f32,%o:f32):
%v = arith.maximumf %x,%o : f32
linalg.yield %v : f32
} -> tensor<1xf32>
%ee = tensor.empty() : tensor<1x512xf32>
%ex = linalg.generic {indexing_maps = [affine_map<(m,n)->(m,n)>,affine_map<(m,n)->(m)>,affine_map<(m,n)->(m,n)>], iterator_types = ["parallel","parallel"]} ins(%scaled,%mx : tensor<1x512xf32>,tensor<1xf32>) outs(%ee : tensor<1x512xf32>) {
^bb0(%x:f32,%m:f32,%o:f32):
%d = arith.subf %x,%m : f32
%e = math.exp %d : f32
linalg.yield %e : f32
} -> tensor<1x512xf32>
%zero = arith.constant dense<0.0> : tensor<1xf32>
%sum = linalg.generic {indexing_maps = [affine_map<(m,n)->(m,n)>,affine_map<(m,n)->(m)>], iterator_types = ["parallel","reduction"]} ins(%ex : tensor<1x512xf32>) outs(%zero : tensor<1xf32>) {
^bb0(%x:f32,%o:f32):
%v = arith.addf %x,%o : f32
linalg.yield %v : f32
} -> tensor<1xf32>
%pe = tensor.empty() : tensor<1x512xf32>
%p = linalg.generic {indexing_maps = [affine_map<(m,n)->(m,n)>,affine_map<(m,n)->(m)>,affine_map<(m,n)->(m,n)>], iterator_types = ["parallel","parallel"]} ins(%ex,%sum : tensor<1x512xf32>,tensor<1xf32>) outs(%pe : tensor<1x512xf32>) {
^bb0(%x:f32,%s:f32,%o:f32):
%v = arith.divf %x,%s : f32
linalg.yield %v : f32
} -> tensor<1x512xf32>
%valz = arith.constant dense<0.0> : tensor<512x128xf32>
%val = linalg.matmul ins(%l,%wv : tensor<512x32xf32>, tensor<32x128xf32>) outs(%valz : tensor<512x128xf32>) -> tensor<512x128xf32>
%rz = arith.constant dense<0.0> : tensor<1x128xf32>
%r = linalg.matmul ins(%p,%val : tensor<1x512xf32>, tensor<512x128xf32>) outs(%rz : tensor<1x128xf32>) -> tensor<1x128xf32>

return %r : tensor<1x128xf32>
}
}
