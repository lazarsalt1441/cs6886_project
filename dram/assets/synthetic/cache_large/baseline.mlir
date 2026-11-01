#map = affine_map<(d0, d1, d2) -> (d0)>
#map1 = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
#map2 = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d3, d4)>
#map3 = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3, d4)>
module @module {
  func.func @main(%arg0: tensor<2x1x8x1x32xf32>, %arg1: tensor<65536x2x32xf32>, %arg2: tensor<65536x2x32xf32>, %arg3: tensor<2x2x2x32xf32>, %arg4: tensor<2x2x2x32xf32>, %arg5: tensor<128xi64>) -> (tensor<1x8x1x32xf32>, tensor<1x8x1x32xf32>) {
    %cst = arith.constant 0.000000e+00 : f32
    %extracted_slice = tensor.extract_slice %arg3[0, 0, 0, 0] [1, 2, 2, 32] [1, 1, 1, 1] : tensor<2x2x2x32xf32> to tensor<1x2x2x32xf32>
    %collapsed = tensor.collapse_shape %extracted_slice [[0, 1], [2], [3]] : tensor<1x2x2x32xf32> into tensor<2x2x32xf32>
    %inserted_slice = tensor.insert_slice %collapsed into %arg1[37, 0, 0] [2, 2, 32] [1, 1, 1] : tensor<2x2x32xf32> into tensor<65536x2x32xf32>
    %extracted_slice_0 = tensor.extract_slice %arg4[0, 0, 0, 0] [1, 2, 2, 32] [1, 1, 1, 1] : tensor<2x2x2x32xf32> to tensor<1x2x2x32xf32>
    %collapsed_1 = tensor.collapse_shape %extracted_slice_0 [[0, 1], [2], [3]] : tensor<1x2x2x32xf32> into tensor<2x2x32xf32>
    %inserted_slice_2 = tensor.insert_slice %collapsed_1 into %arg2[37, 0, 0] [2, 2, 32] [1, 1, 1] : tensor<2x2x32xf32> into tensor<65536x2x32xf32>
    %0 = tensor.empty() : tensor<128x2x32xf32>
    %1 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%arg5 : tensor<128xi64>) outs(%0 : tensor<128x2x32xf32>) {
    ^bb0(%in: i64, %out: f32):
      %15 = arith.index_cast %in : i64 to index
      %16 = linalg.index 1 : index
      %17 = linalg.index 2 : index
      %extracted = tensor.extract %inserted_slice[%15, %16, %17] : tensor<65536x2x32xf32>
      linalg.yield %extracted : f32
    } -> tensor<128x2x32xf32>
    %2 = tensor.empty() : tensor<2x128x32xf32>
    %transposed = linalg.transpose ins(%1 : tensor<128x2x32xf32>) outs(%2 : tensor<2x128x32xf32>) permutation = [1, 0, 2] 
    %3 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%arg5 : tensor<128xi64>) outs(%0 : tensor<128x2x32xf32>) {
    ^bb0(%in: i64, %out: f32):
      %15 = arith.index_cast %in : i64 to index
      %16 = linalg.index 1 : index
      %17 = linalg.index 2 : index
      %extracted = tensor.extract %inserted_slice_2[%15, %16, %17] : tensor<65536x2x32xf32>
      linalg.yield %extracted : f32
    } -> tensor<128x2x32xf32>
    %transposed_3 = linalg.transpose ins(%3 : tensor<128x2x32xf32>) outs(%2 : tensor<2x128x32xf32>) permutation = [1, 0, 2] 
    %extracted_slice_4 = tensor.extract_slice %arg0[0, 0, 0, 0, 0] [1, 1, 8, 1, 32] [1, 1, 1, 1, 1] : tensor<2x1x8x1x32xf32> to tensor<1x1x8x1x32xf32>
    %expanded = tensor.expand_shape %transposed [[0, 1], [2, 3], [4]] output_shape [1, 2, 1, 128, 32] : tensor<2x128x32xf32> into tensor<1x2x1x128x32xf32>
    %4 = tensor.empty() : tensor<1x2x4x128x32xf32>
    %collapsed_5 = tensor.collapse_shape %expanded [[0], [1, 2], [3], [4]] : tensor<1x2x1x128x32xf32> into tensor<1x2x128x32xf32>
    %5 = linalg.generic {indexing_maps = [#map2, #map3], iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel"]} ins(%collapsed_5 : tensor<1x2x128x32xf32>) outs(%4 : tensor<1x2x4x128x32xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x2x4x128x32xf32>
    %expanded_6 = tensor.expand_shape %transposed_3 [[0, 1], [2, 3], [4]] output_shape [1, 2, 1, 128, 32] : tensor<2x128x32xf32> into tensor<1x2x1x128x32xf32>
    %collapsed_7 = tensor.collapse_shape %expanded_6 [[0], [1, 2], [3], [4]] : tensor<1x2x1x128x32xf32> into tensor<1x2x128x32xf32>
    %6 = linalg.generic {indexing_maps = [#map2, #map3], iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel"]} ins(%collapsed_7 : tensor<1x2x128x32xf32>) outs(%4 : tensor<1x2x4x128x32xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x2x4x128x32xf32>
    %collapsed_8 = tensor.collapse_shape %extracted_slice_4 [[0, 1, 2], [3], [4]] : tensor<1x1x8x1x32xf32> into tensor<8x1x32xf32>
    %collapsed_9 = tensor.collapse_shape %5 [[0, 1, 2], [3], [4]] : tensor<1x2x4x128x32xf32> into tensor<8x128x32xf32>
    %collapsed_10 = tensor.collapse_shape %6 [[0, 1, 2], [3], [4]] : tensor<1x2x4x128x32xf32> into tensor<8x128x32xf32>
    %7 = tensor.empty() : tensor<8x1x32xf32>
    %8 = linalg.fill ins(%cst : f32) outs(%7 : tensor<8x1x32xf32>) -> tensor<8x1x32xf32>
    %9 = tm_tensor.attention ins(%collapsed_8, %collapsed_9, %collapsed_10 : tensor<8x1x32xf32>, tensor<8x128x32xf32>, tensor<8x128x32xf32>) outs(%8 : tensor<8x1x32xf32>) -> tensor<8x1x32xf32>
    %expanded_11 = tensor.expand_shape %9 [[0, 1], [2], [3]] output_shape [1, 8, 1, 32] : tensor<8x1x32xf32> into tensor<1x8x1x32xf32>
    %extracted_slice_12 = tensor.extract_slice %arg3[1, 0, 0, 0] [1, 2, 2, 32] [1, 1, 1, 1] : tensor<2x2x2x32xf32> to tensor<1x2x2x32xf32>
    %collapsed_13 = tensor.collapse_shape %extracted_slice_12 [[0, 1], [2], [3]] : tensor<1x2x2x32xf32> into tensor<2x2x32xf32>
    %inserted_slice_14 = tensor.insert_slice %collapsed_13 into %arg1[37, 0, 0] [2, 2, 32] [1, 1, 1] : tensor<2x2x32xf32> into tensor<65536x2x32xf32>
    %extracted_slice_15 = tensor.extract_slice %arg4[1, 0, 0, 0] [1, 2, 2, 32] [1, 1, 1, 1] : tensor<2x2x2x32xf32> to tensor<1x2x2x32xf32>
    %collapsed_16 = tensor.collapse_shape %extracted_slice_15 [[0, 1], [2], [3]] : tensor<1x2x2x32xf32> into tensor<2x2x32xf32>
    %inserted_slice_17 = tensor.insert_slice %collapsed_16 into %arg2[37, 0, 0] [2, 2, 32] [1, 1, 1] : tensor<2x2x32xf32> into tensor<65536x2x32xf32>
    %10 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%arg5 : tensor<128xi64>) outs(%0 : tensor<128x2x32xf32>) {
    ^bb0(%in: i64, %out: f32):
      %15 = arith.index_cast %in : i64 to index
      %16 = linalg.index 1 : index
      %17 = linalg.index 2 : index
      %extracted = tensor.extract %inserted_slice_14[%15, %16, %17] : tensor<65536x2x32xf32>
      linalg.yield %extracted : f32
    } -> tensor<128x2x32xf32>
    %transposed_18 = linalg.transpose ins(%10 : tensor<128x2x32xf32>) outs(%2 : tensor<2x128x32xf32>) permutation = [1, 0, 2] 
    %11 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%arg5 : tensor<128xi64>) outs(%0 : tensor<128x2x32xf32>) {
    ^bb0(%in: i64, %out: f32):
      %15 = arith.index_cast %in : i64 to index
      %16 = linalg.index 1 : index
      %17 = linalg.index 2 : index
      %extracted = tensor.extract %inserted_slice_17[%15, %16, %17] : tensor<65536x2x32xf32>
      linalg.yield %extracted : f32
    } -> tensor<128x2x32xf32>
    %transposed_19 = linalg.transpose ins(%11 : tensor<128x2x32xf32>) outs(%2 : tensor<2x128x32xf32>) permutation = [1, 0, 2] 
    %extracted_slice_20 = tensor.extract_slice %arg0[1, 0, 0, 0, 0] [1, 1, 8, 1, 32] [1, 1, 1, 1, 1] : tensor<2x1x8x1x32xf32> to tensor<1x1x8x1x32xf32>
    %expanded_21 = tensor.expand_shape %transposed_18 [[0, 1], [2, 3], [4]] output_shape [1, 2, 1, 128, 32] : tensor<2x128x32xf32> into tensor<1x2x1x128x32xf32>
    %collapsed_22 = tensor.collapse_shape %expanded_21 [[0], [1, 2], [3], [4]] : tensor<1x2x1x128x32xf32> into tensor<1x2x128x32xf32>
    %12 = linalg.generic {indexing_maps = [#map2, #map3], iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel"]} ins(%collapsed_22 : tensor<1x2x128x32xf32>) outs(%4 : tensor<1x2x4x128x32xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x2x4x128x32xf32>
    %expanded_23 = tensor.expand_shape %transposed_19 [[0, 1], [2, 3], [4]] output_shape [1, 2, 1, 128, 32] : tensor<2x128x32xf32> into tensor<1x2x1x128x32xf32>
    %collapsed_24 = tensor.collapse_shape %expanded_23 [[0], [1, 2], [3], [4]] : tensor<1x2x1x128x32xf32> into tensor<1x2x128x32xf32>
    %13 = linalg.generic {indexing_maps = [#map2, #map3], iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel"]} ins(%collapsed_24 : tensor<1x2x128x32xf32>) outs(%4 : tensor<1x2x4x128x32xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x2x4x128x32xf32>
    %collapsed_25 = tensor.collapse_shape %extracted_slice_20 [[0, 1, 2], [3], [4]] : tensor<1x1x8x1x32xf32> into tensor<8x1x32xf32>
    %collapsed_26 = tensor.collapse_shape %12 [[0, 1, 2], [3], [4]] : tensor<1x2x4x128x32xf32> into tensor<8x128x32xf32>
    %collapsed_27 = tensor.collapse_shape %13 [[0, 1, 2], [3], [4]] : tensor<1x2x4x128x32xf32> into tensor<8x128x32xf32>
    %14 = tm_tensor.attention ins(%collapsed_25, %collapsed_26, %collapsed_27 : tensor<8x1x32xf32>, tensor<8x128x32xf32>, tensor<8x128x32xf32>) outs(%8 : tensor<8x1x32xf32>) -> tensor<8x1x32xf32>
    %expanded_28 = tensor.expand_shape %14 [[0, 1], [2], [3]] output_shape [1, 8, 1, 32] : tensor<8x1x32xf32> into tensor<1x8x1x32xf32>
    return %expanded_11, %expanded_28 : tensor<1x8x1x32xf32>, tensor<1x8x1x32xf32>
  }
}

