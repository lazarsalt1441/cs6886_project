module {func.func @main(%history: tensor<1x2x8192x4xf32>, %other: tensor<1x2x8192x4xf32>, %a: tensor<1x2x1x4xf32>, %b: tensor<1x2x1x4xf32>, %c: tensor<i1>) -> tensor<1x2x8193x4xf32> {
 %c2 = arith.constant 2 : index
 %cp = arith.constant 1 : index
 %x = tensor.concat dim(2) %history, %a : (tensor<1x2x8192x4xf32>, tensor<1x2x1x4xf32>) -> tensor<1x2x8193x4xf32>
 %y = tensor.concat dim(2) %history, %b : (tensor<1x2x8192x4xf32>, tensor<1x2x1x4xf32>) -> tensor<1x2x8193x4xf32>
 %e = tensor.empty() : tensor<1x2x8193x4xf32>
 %r = linalg.generic {indexing_maps=[affine_map<(d0,d1,d2,d3)->()>, affine_map<(d0,d1,d2,d3)->(d0,d1,d2,d3)>, affine_map<(d0,d1,d2,d3)->(d0,d1,d2,d3)>, affine_map<(d0,d1,d2,d3)->(d0,d1,d2,d3)>], iterator_types=["parallel","parallel","parallel","parallel"]}
 ins(%c,%x,%y : tensor<i1>,tensor<1x2x8193x4xf32>,tensor<1x2x8193x4xf32>) outs(%e : tensor<1x2x8193x4xf32>) {
 ^bb0(%c0: i1,%v: f32,%w: f32,%o: f32):
 %z = arith.select %c0,%v,%w : f32
 linalg.yield %z : f32
 } -> tensor<1x2x8193x4xf32>
 return %r : tensor<1x2x8193x4xf32>
}}