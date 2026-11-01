#map = affine_map<(d0, d1, d2) -> (d0)>
#map1 = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
#map2 = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d3, d4)>
#map3 = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3, d4)>
module @module {
  func.func @main(%arg0: tensor<1x1x8x1x32xf32>, %arg1: tensor<4096x2x32xf32>, %arg2: tensor<4096x2x32xf32>, %arg3: tensor<1x2x2x32xf32>, %arg4: tensor<1x2x2x32xf32>, %arg5: tensor<128xi64>) -> tensor<1x8x1x32xf32> {
    %cst = arith.constant 0.000000e+00 : f32
    %collapsed = tensor.collapse_shape %arg3 [[0, 1], [2], [3]] : tensor<1x2x2x32xf32> into tensor<2x2x32xf32>
    %inserted_slice = tensor.insert_slice %collapsed into %arg1[37, 0, 0] [2, 2, 32] [1, 1, 1] : tensor<2x2x32xf32> into tensor<4096x2x32xf32>
    %collapsed_0 = tensor.collapse_shape %arg4 [[0, 1], [2], [3]] : tensor<1x2x2x32xf32> into tensor<2x2x32xf32>
    %inserted_slice_1 = tensor.insert_slice %collapsed_0 into %arg2[37, 0, 0] [2, 2, 32] [1, 1, 1] : tensor<2x2x32xf32> into tensor<4096x2x32xf32>
    %0 = tensor.empty() : tensor<128x2x32xf32>
    %1 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%arg5 : tensor<128xi64>) outs(%0 : tensor<128x2x32xf32>) {
    ^bb0(%in: i64, %out: f32):
      %10 = arith.index_cast %in : i64 to index
      %11 = linalg.index 1 : index
      %12 = linalg.index 2 : index
      %extracted = tensor.extract %inserted_slice[%10, %11, %12] : tensor<4096x2x32xf32>
      linalg.yield %extracted : f32
    } -> tensor<128x2x32xf32>
    %2 = tensor.empty() : tensor<2x128x32xf32>
    %transposed = linalg.transpose ins(%1 : tensor<128x2x32xf32>) outs(%2 : tensor<2x128x32xf32>) permutation = [1, 0, 2] 
    %3 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%arg5 : tensor<128xi64>) outs(%0 : tensor<128x2x32xf32>) {
    ^bb0(%in: i64, %out: f32):
      %10 = arith.index_cast %in : i64 to index
      %11 = linalg.index 1 : index
      %12 = linalg.index 2 : index
      %extracted = tensor.extract %inserted_slice_1[%10, %11, %12] : tensor<4096x2x32xf32>
      linalg.yield %extracted : f32
    } -> tensor<128x2x32xf32>
    %transposed_2 = linalg.transpose ins(%3 : tensor<128x2x32xf32>) outs(%2 : tensor<2x128x32xf32>) permutation = [1, 0, 2] 
    %expanded = tensor.expand_shape %transposed [[0, 1], [2, 3], [4]] output_shape [1, 2, 1, 128, 32] : tensor<2x128x32xf32> into tensor<1x2x1x128x32xf32>
    %4 = tensor.empty() : tensor<1x2x4x128x32xf32>
    %collapsed_3 = tensor.collapse_shape %expanded [[0], [1, 2], [3], [4]] : tensor<1x2x1x128x32xf32> into tensor<1x2x128x32xf32>
    %5 = linalg.generic {indexing_maps = [#map2, #map3], iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel"]} ins(%collapsed_3 : tensor<1x2x128x32xf32>) outs(%4 : tensor<1x2x4x128x32xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x2x4x128x32xf32>
    %expanded_4 = tensor.expand_shape %transposed_2 [[0, 1], [2, 3], [4]] output_shape [1, 2, 1, 128, 32] : tensor<2x128x32xf32> into tensor<1x2x1x128x32xf32>
    %collapsed_5 = tensor.collapse_shape %expanded_4 [[0], [1, 2], [3], [4]] : tensor<1x2x1x128x32xf32> into tensor<1x2x128x32xf32>
    %6 = linalg.generic {indexing_maps = [#map2, #map3], iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel"]} ins(%collapsed_5 : tensor<1x2x128x32xf32>) outs(%4 : tensor<1x2x4x128x32xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x2x4x128x32xf32>
    %collapsed_6 = tensor.collapse_shape %arg0 [[0, 1, 2], [3], [4]] : tensor<1x1x8x1x32xf32> into tensor<8x1x32xf32>
    %collapsed_7 = tensor.collapse_shape %5 [[0, 1, 2], [3], [4]] : tensor<1x2x4x128x32xf32> into tensor<8x128x32xf32>
    %collapsed_8 = tensor.collapse_shape %6 [[0, 1, 2], [3], [4]] : tensor<1x2x4x128x32xf32> into tensor<8x128x32xf32>
    %7 = tensor.empty() : tensor<8x1x32xf32>
    %8 = linalg.fill ins(%cst : f32) outs(%7 : tensor<8x1x32xf32>) -> tensor<8x1x32xf32>
    %9 = tm_tensor.attention ins(%collapsed_6, %collapsed_7, %collapsed_8 : tensor<8x1x32xf32>, tensor<8x128x32xf32>, tensor<8x128x32xf32>) outs(%8 : tensor<8x1x32xf32>) -> tensor<8x1x32xf32>
    %expanded_9 = tensor.expand_shape %9 [[0, 1], [2], [3]] output_shape [1, 8, 1, 32] : tensor<8x1x32xf32> into tensor<1x8x1x32xf32>
    return %expanded_9 : tensor<1x8x1x32xf32>
  }
}

