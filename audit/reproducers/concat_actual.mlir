#map = affine_map<(d0, d1, d2, d3) -> ()>
#map1 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
module {
  func.func @main(%arg0: tensor<1x2x8192x4xf32>, %arg1: tensor<1x2x8192x4xf32>, %arg2: tensor<1x2x1x4xf32>, %arg3: tensor<1x2x1x4xf32>, %arg4: tensor<i1>) -> tensor<1x2x8193x4xf32> {
    %0 = tensor.empty() : tensor<1x2x1x4xf32>
    %1 = linalg.generic {indexing_maps = [#map, #map1, #map1, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%arg4, %arg2, %arg3 : tensor<i1>, tensor<1x2x1x4xf32>, tensor<1x2x1x4xf32>) outs(%0 : tensor<1x2x1x4xf32>) {
    ^bb0(%in: i1, %in_0: f32, %in_1: f32, %out: f32):
      %2 = arith.select %in, %in_0, %in_1 : f32
      linalg.yield %2 : f32
    } -> tensor<1x2x1x4xf32>
    %concat = tensor.concat dim(2) %arg0, %1 {attention.concat_commit_factored} : (tensor<1x2x8192x4xf32>, tensor<1x2x1x4xf32>) -> tensor<1x2x8193x4xf32>
    return %concat : tensor<1x2x8193x4xf32>
  }
}

