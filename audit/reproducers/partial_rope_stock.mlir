module {
func.func @main(%q: tensor<1x16xf32>, %l: tensor<32x8xf32>, %wk: tensor<8x16xf32>, %qr: tensor<1x4xf32>, %kr: tensor<32x4xf32>, %wv: tensor<8x16xf32>) -> tensor<1x16xf32> {
%keyz = arith.constant dense<0.0> : tensor<32x16xf32>
%key = linalg.matmul ins(%l,%wk : tensor<32x8xf32>, tensor<8x16xf32>) outs(%keyz : tensor<32x16xf32>) -> tensor<32x16xf32>
%contentz = arith.constant dense<0.0> : tensor<1x32xf32>
%content = linalg.matmul indexing_maps = [affine_map<(m,n,k)->(m,k)>,affine_map<(m,n,k)->(n,k)>,affine_map<(m,n,k)->(m,n)>] ins(%q,%key : tensor<1x16xf32>, tensor<32x16xf32>) outs(%contentz : tensor<1x32xf32>) -> tensor<1x32xf32>
%positionz = arith.constant dense<0.0> : tensor<1x32xf32>
%position = linalg.matmul indexing_maps = [affine_map<(m,n,k)->(m,k)>,affine_map<(m,n,k)->(n,k)>,affine_map<(m,n,k)->(m,n)>] ins(%qr,%kr : tensor<1x4xf32>, tensor<32x4xf32>) outs(%positionz : tensor<1x32xf32>) -> tensor<1x32xf32>
%score = arith.addf %content,%position : tensor<1x32xf32>
%scale = arith.constant dense<2.236067977500e-01> : tensor<1x32xf32>
%scaled = arith.mulf %score, %scale : tensor<1x32xf32>
%neg = arith.constant dense<0xFF800000> : tensor<1xf32>
%mx = linalg.generic {indexing_maps = [affine_map<(m,n)->(m,n)>,affine_map<(m,n)->(m)>], iterator_types = ["parallel","reduction"]} ins(%scaled : tensor<1x32xf32>) outs(%neg : tensor<1xf32>) {
^bb0(%x:f32,%o:f32):
%v = arith.maximumf %x,%o : f32
linalg.yield %v : f32
} -> tensor<1xf32>
%ee = tensor.empty() : tensor<1x32xf32>
%ex = linalg.generic {indexing_maps = [affine_map<(m,n)->(m,n)>,affine_map<(m,n)->(m)>,affine_map<(m,n)->(m,n)>], iterator_types = ["parallel","parallel"]} ins(%scaled,%mx : tensor<1x32xf32>,tensor<1xf32>) outs(%ee : tensor<1x32xf32>) {
^bb0(%x:f32,%m:f32,%o:f32):
%d = arith.subf %x,%m : f32
%e = math.exp %d : f32
linalg.yield %e : f32
} -> tensor<1x32xf32>
%zero = arith.constant dense<0.0> : tensor<1xf32>
%sum = linalg.generic {indexing_maps = [affine_map<(m,n)->(m,n)>,affine_map<(m,n)->(m)>], iterator_types = ["parallel","reduction"]} ins(%ex : tensor<1x32xf32>) outs(%zero : tensor<1xf32>) {
^bb0(%x:f32,%o:f32):
%v = arith.addf %x,%o : f32
linalg.yield %v : f32
} -> tensor<1xf32>
%pe = tensor.empty() : tensor<1x32xf32>
%p = linalg.generic {indexing_maps = [affine_map<(m,n)->(m,n)>,affine_map<(m,n)->(m)>,affine_map<(m,n)->(m,n)>], iterator_types = ["parallel","parallel"]} ins(%ex,%sum : tensor<1x32xf32>,tensor<1xf32>) outs(%pe : tensor<1x32xf32>) {
^bb0(%x:f32,%s:f32,%o:f32):
%v = arith.divf %x,%s : f32
linalg.yield %v : f32
} -> tensor<1x32xf32>
%valz = arith.constant dense<0.0> : tensor<32x16xf32>
%val = linalg.matmul ins(%l,%wv : tensor<32x8xf32>, tensor<8x16xf32>) outs(%valz : tensor<32x16xf32>) -> tensor<32x16xf32>
%rz = arith.constant dense<0.0> : tensor<1x16xf32>
%r = linalg.matmul ins(%p,%val : tensor<1x32xf32>, tensor<32x16xf32>) outs(%rz : tensor<1x16xf32>) -> tensor<1x16xf32>

return %r : tensor<1x16xf32>
}
}
