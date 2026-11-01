#map = affine_map<(d0, d1, d2) -> (d0)>
#map1 = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
#map2 = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d3, d4)>
#map3 = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3, d4)>
module @module {
  func.func @main(%arg0: tensor<2x1x8x1x32xf32>, %arg1: tensor<65536x2x32xf32>, %arg2: tensor<65536x2x32xf32>, %arg3: tensor<2x2x2x32xf32>, %arg4: tensor<2x2x2x32xf32>, %arg5: tensor<128xi64>) -> (tensor<1x8x1x32xf32>, tensor<1x8x1x32xf32>) {
    %c2 = arith.constant 2 : index
    %c37 = arith.constant 37 : index
    %cst = arith.constant 0.000000e+00 : f32
    %extracted_slice = tensor.extract_slice %arg3[0, 0, 0, 0] [1, 2, 2, 32] [1, 1, 1, 1] : tensor<2x2x2x32xf32> to tensor<1x2x2x32xf32>
    %collapsed = tensor.collapse_shape %extracted_slice [[0, 1], [2], [3]] : tensor<1x2x2x32xf32> into tensor<2x2x32xf32>
    %extracted_slice_0 = tensor.extract_slice %arg4[0, 0, 0, 0] [1, 2, 2, 32] [1, 1, 1, 1] : tensor<2x2x2x32xf32> to tensor<1x2x2x32xf32>
    %collapsed_1 = tensor.collapse_shape %extracted_slice_0 [[0, 1], [2], [3]] : tensor<1x2x2x32xf32> into tensor<2x2x32xf32>
    %0 = tensor.empty() : tensor<128x2x32xf32>
    %1 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%arg5 : tensor<128xi64>) outs(%0 : tensor<128x2x32xf32>) {
    ^bb0(%in: i64, %out: f32):
      %15 = arith.index_cast %in : i64 to index
      %16 = linalg.index 1 : index
      %17 = linalg.index 2 : index
      %18 = arith.subi %15, %c37 : index
      %19 = arith.cmpi ult, %18, %c2 : index
      %20 = scf.if %19 -> (f32) {
        %extracted = tensor.extract %collapsed[%18, %16, %17] : tensor<2x2x32xf32>
        scf.yield %extracted : f32
      } else {
        %extracted = tensor.extract %arg1[%15, %16, %17] : tensor<65536x2x32xf32>
        scf.yield %extracted : f32
      } {attention.cache_read_forwarded}
      linalg.yield %20 : f32
    } -> tensor<128x2x32xf32>
    %2 = tensor.empty() : tensor<2x128x32xf32>
    %transposed = linalg.transpose ins(%1 : tensor<128x2x32xf32>) outs(%2 : tensor<2x128x32xf32>) permutation = [1, 0, 2] 
    %3 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%arg5 : tensor<128xi64>) outs(%0 : tensor<128x2x32xf32>) {
    ^bb0(%in: i64, %out: f32):
      %15 = arith.index_cast %in : i64 to index
      %16 = linalg.index 1 : index
      %17 = linalg.index 2 : index
      %18 = arith.subi %15, %c37 : index
      %19 = arith.cmpi ult, %18, %c2 : index
      %20 = scf.if %19 -> (f32) {
        %extracted = tensor.extract %collapsed_1[%18, %16, %17] : tensor<2x2x32xf32>
        scf.yield %extracted : f32
      } else {
        %extracted = tensor.extract %arg2[%15, %16, %17] : tensor<65536x2x32xf32>
        scf.yield %extracted : f32
      } {attention.cache_read_forwarded}
      linalg.yield %20 : f32
    } -> tensor<128x2x32xf32>
    %transposed_2 = linalg.transpose ins(%3 : tensor<128x2x32xf32>) outs(%2 : tensor<2x128x32xf32>) permutation = [1, 0, 2] 
    %extracted_slice_3 = tensor.extract_slice %arg0[0, 0, 0, 0, 0] [1, 1, 8, 1, 32] [1, 1, 1, 1, 1] : tensor<2x1x8x1x32xf32> to tensor<1x1x8x1x32xf32>
    %expanded = tensor.expand_shape %transposed [[0, 1], [2, 3], [4]] output_shape [1, 2, 1, 128, 32] : tensor<2x128x32xf32> into tensor<1x2x1x128x32xf32>
    %4 = tensor.empty() : tensor<1x2x4x128x32xf32>
    %collapsed_4 = tensor.collapse_shape %expanded [[0], [1, 2], [3], [4]] : tensor<1x2x1x128x32xf32> into tensor<1x2x128x32xf32>
    %5 = linalg.generic {indexing_maps = [#map2, #map3], iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel"]} ins(%collapsed_4 : tensor<1x2x128x32xf32>) outs(%4 : tensor<1x2x4x128x32xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x2x4x128x32xf32>
    %expanded_5 = tensor.expand_shape %transposed_2 [[0, 1], [2, 3], [4]] output_shape [1, 2, 1, 128, 32] : tensor<2x128x32xf32> into tensor<1x2x1x128x32xf32>
    %collapsed_6 = tensor.collapse_shape %expanded_5 [[0], [1, 2], [3], [4]] : tensor<1x2x1x128x32xf32> into tensor<1x2x128x32xf32>
    %6 = linalg.generic {indexing_maps = [#map2, #map3], iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel"]} ins(%collapsed_6 : tensor<1x2x128x32xf32>) outs(%4 : tensor<1x2x4x128x32xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x2x4x128x32xf32>
    %collapsed_7 = tensor.collapse_shape %extracted_slice_3 [[0, 1, 2], [3], [4]] : tensor<1x1x8x1x32xf32> into tensor<8x1x32xf32>
    %collapsed_8 = tensor.collapse_shape %5 [[0, 1, 2], [3], [4]] : tensor<1x2x4x128x32xf32> into tensor<8x128x32xf32>
    %collapsed_9 = tensor.collapse_shape %6 [[0, 1, 2], [3], [4]] : tensor<1x2x4x128x32xf32> into tensor<8x128x32xf32>
    %7 = tensor.empty() : tensor<8x1x32xf32>
    %8 = linalg.fill ins(%cst : f32) outs(%7 : tensor<8x1x32xf32>) -> tensor<8x1x32xf32>
    %9 = "tm_tensor.attention"(%collapsed_7, %collapsed_8, %collapsed_9, %8) <{operandSegmentSizes = array<i32: 3, 1>}> : (tensor<8x1x32xf32>, tensor<8x128x32xf32>, tensor<8x128x32xf32>, tensor<8x1x32xf32>) -> tensor<8x1x32xf32>
    %expanded_10 = tensor.expand_shape %9 [[0, 1], [2], [3]] output_shape [1, 8, 1, 32] : tensor<8x1x32xf32> into tensor<1x8x1x32xf32>
    %extracted_slice_11 = tensor.extract_slice %arg3[1, 0, 0, 0] [1, 2, 2, 32] [1, 1, 1, 1] : tensor<2x2x2x32xf32> to tensor<1x2x2x32xf32>
    %collapsed_12 = tensor.collapse_shape %extracted_slice_11 [[0, 1], [2], [3]] : tensor<1x2x2x32xf32> into tensor<2x2x32xf32>
    %extracted_slice_13 = tensor.extract_slice %arg4[1, 0, 0, 0] [1, 2, 2, 32] [1, 1, 1, 1] : tensor<2x2x2x32xf32> to tensor<1x2x2x32xf32>
    %collapsed_14 = tensor.collapse_shape %extracted_slice_13 [[0, 1], [2], [3]] : tensor<1x2x2x32xf32> into tensor<2x2x32xf32>
    %10 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%arg5 : tensor<128xi64>) outs(%0 : tensor<128x2x32xf32>) {
    ^bb0(%in: i64, %out: f32):
      %15 = arith.index_cast %in : i64 to index
      %16 = linalg.index 1 : index
      %17 = linalg.index 2 : index
      %18 = arith.subi %15, %c37 : index
      %19 = arith.cmpi ult, %18, %c2 : index
      %20 = scf.if %19 -> (f32) {
        %extracted = tensor.extract %collapsed_12[%18, %16, %17] : tensor<2x2x32xf32>
        scf.yield %extracted : f32
      } else {
        %extracted = tensor.extract %arg1[%15, %16, %17] : tensor<65536x2x32xf32>
        scf.yield %extracted : f32
      } {attention.cache_read_forwarded}
      linalg.yield %20 : f32
    } -> tensor<128x2x32xf32>
    %transposed_15 = linalg.transpose ins(%10 : tensor<128x2x32xf32>) outs(%2 : tensor<2x128x32xf32>) permutation = [1, 0, 2] 
    %11 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%arg5 : tensor<128xi64>) outs(%0 : tensor<128x2x32xf32>) {
    ^bb0(%in: i64, %out: f32):
      %15 = arith.index_cast %in : i64 to index
      %16 = linalg.index 1 : index
      %17 = linalg.index 2 : index
      %18 = arith.subi %15, %c37 : index
      %19 = arith.cmpi ult, %18, %c2 : index
      %20 = scf.if %19 -> (f32) {
        %extracted = tensor.extract %collapsed_14[%18, %16, %17] : tensor<2x2x32xf32>
        scf.yield %extracted : f32
      } else {
        %extracted = tensor.extract %arg2[%15, %16, %17] : tensor<65536x2x32xf32>
        scf.yield %extracted : f32
      } {attention.cache_read_forwarded}
      linalg.yield %20 : f32
    } -> tensor<128x2x32xf32>
    %transposed_16 = linalg.transpose ins(%11 : tensor<128x2x32xf32>) outs(%2 : tensor<2x128x32xf32>) permutation = [1, 0, 2] 
    %extracted_slice_17 = tensor.extract_slice %arg0[1, 0, 0, 0, 0] [1, 1, 8, 1, 32] [1, 1, 1, 1, 1] : tensor<2x1x8x1x32xf32> to tensor<1x1x8x1x32xf32>
    %expanded_18 = tensor.expand_shape %transposed_15 [[0, 1], [2, 3], [4]] output_shape [1, 2, 1, 128, 32] : tensor<2x128x32xf32> into tensor<1x2x1x128x32xf32>
    %collapsed_19 = tensor.collapse_shape %expanded_18 [[0], [1, 2], [3], [4]] : tensor<1x2x1x128x32xf32> into tensor<1x2x128x32xf32>
    %12 = linalg.generic {indexing_maps = [#map2, #map3], iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel"]} ins(%collapsed_19 : tensor<1x2x128x32xf32>) outs(%4 : tensor<1x2x4x128x32xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x2x4x128x32xf32>
    %expanded_20 = tensor.expand_shape %transposed_16 [[0, 1], [2, 3], [4]] output_shape [1, 2, 1, 128, 32] : tensor<2x128x32xf32> into tensor<1x2x1x128x32xf32>
    %collapsed_21 = tensor.collapse_shape %expanded_20 [[0], [1, 2], [3], [4]] : tensor<1x2x1x128x32xf32> into tensor<1x2x128x32xf32>
    %13 = linalg.generic {indexing_maps = [#map2, #map3], iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel"]} ins(%collapsed_21 : tensor<1x2x128x32xf32>) outs(%4 : tensor<1x2x4x128x32xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x2x4x128x32xf32>
    %collapsed_22 = tensor.collapse_shape %extracted_slice_17 [[0, 1, 2], [3], [4]] : tensor<1x1x8x1x32xf32> into tensor<8x1x32xf32>
    %collapsed_23 = tensor.collapse_shape %12 [[0, 1, 2], [3], [4]] : tensor<1x2x4x128x32xf32> into tensor<8x128x32xf32>
    %collapsed_24 = tensor.collapse_shape %13 [[0, 1, 2], [3], [4]] : tensor<1x2x4x128x32xf32> into tensor<8x128x32xf32>
    %14 = "tm_tensor.attention"(%collapsed_22, %collapsed_23, %collapsed_24, %8) <{operandSegmentSizes = array<i32: 3, 1>}> : (tensor<8x1x32xf32>, tensor<8x128x32xf32>, tensor<8x128x32xf32>, tensor<8x1x32xf32>) -> tensor<8x1x32xf32>
    %expanded_25 = tensor.expand_shape %14 [[0, 1], [2], [3]] output_shape [1, 8, 1, 32] : tensor<8x1x32xf32> into tensor<1x8x1x32xf32>
    return %expanded_10, %expanded_25 : tensor<1x8x1x32xf32>, tensor<1x8x1x32xf32>
  }
}

