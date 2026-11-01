#map = affine_map<(d0, d1) -> (d0)>
#map1 = affine_map<(d0, d1) -> (d0, d1)>
module {
  func.func @main(%arg0: tensor<512x16xf32>, %arg1: tensor<4x16xf32>, %arg2: tensor<4x16xf32>, %arg3: tensor<8xi64>) -> tensor<16x16xf32> {
    %concat = tensor.concat dim(0) %arg1, %arg1 : (tensor<4x16xf32>, tensor<4x16xf32>) -> tensor<8x16xf32>
    %inserted_slice = tensor.insert_slice %concat into %arg0[128, 0] [8, 16] [1, 1] {attention.cache_updates_coalesced} : tensor<8x16xf32> into tensor<512x16xf32>
    %inserted_slice_0 = tensor.insert_slice %arg2 into %arg0[256, 0] [4, 16] [1, 1] : tensor<4x16xf32> into tensor<512x16xf32>
    %0 = tensor.empty() : tensor<8x16xf32>
    %1 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "parallel"]} ins(%arg3 : tensor<8xi64>) outs(%0 : tensor<8x16xf32>) {
    ^bb0(%in: i64, %out: f32):
      %4 = arith.index_cast %in : i64 to index
      %5 = linalg.index 1 : index
      %extracted = tensor.extract %inserted_slice[%4, %5] : tensor<512x16xf32>
      linalg.yield %extracted : f32
    } -> tensor<8x16xf32>
    %2 = tensor.empty() : tensor<8x16xf32>
    %3 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "parallel"]} ins(%arg3 : tensor<8xi64>) outs(%2 : tensor<8x16xf32>) {
    ^bb0(%in: i64, %out: f32):
      %4 = arith.index_cast %in : i64 to index
      %5 = linalg.index 1 : index
      %extracted = tensor.extract %inserted_slice_0[%4, %5] : tensor<512x16xf32>
      linalg.yield %extracted : f32
    } -> tensor<8x16xf32>
    %concat_1 = tensor.concat dim(0) %1, %3 : (tensor<8x16xf32>, tensor<8x16xf32>) -> tensor<16x16xf32>
    return %concat_1 : tensor<16x16xf32>
  }
}

