#map = affine_map<(d0) -> ()>
#map1 = affine_map<(d0) -> (d0)>
module {
  func.func @main(%arg0: tensor<4096xf32>, %arg1: tensor<16xf32>, %arg2: tensor<16xf32>, %arg3: tensor<i1>) -> tensor<4096xf32> {
    %0 = tensor.empty() : tensor<16xf32>
    %1 = linalg.generic {indexing_maps = [#map, #map1, #map1, #map1], iterator_types = ["parallel"]} ins(%arg3, %arg1, %arg2 : tensor<i1>, tensor<16xf32>, tensor<16xf32>) outs(%0 : tensor<16xf32>) {
    ^bb0(%in: i1, %in_0: f32, %in_1: f32, %out: f32):
      %2 = arith.select %in, %in_0, %in_1 : f32
      linalg.yield %2 : f32
    } -> tensor<16xf32>
    %inserted_slice = tensor.insert_slice %1 into %arg0[128] [16] [1] {attention.commit_factored} : tensor<16xf32> into tensor<4096xf32>
    return %inserted_slice : tensor<4096xf32>
  }
}

