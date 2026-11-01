#map = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>
#map1 = affine_map<(d0, d1, d2, d3) -> (0, d2, d3)>
#map2 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>
module {
  func.func @main(%arg0: tensor<1x1x20xf32>, %arg1: tensor<1x32x8xf32>, %arg2: tensor<1x8x16xf32>, %arg3: tensor<1x32x4xf32>) -> tensor<1x1x32xf32> attributes {attention.allow_reassociation} {
    %cst = arith.constant 0.000000e+00 : f32
    %extracted_slice = tensor.extract_slice %arg0[0, 0, 0] [1, 1, 16] [1, 1, 1] : tensor<1x1x20xf32> to tensor<1x1x16xf32>
    %extracted_slice_0 = tensor.extract_slice %arg0[0, 0, 16] [1, 1, 4] [1, 1, 1] : tensor<1x1x20xf32> to tensor<1x1x4xf32>
    %0 = tensor.empty() : tensor<1x16x8xf32>
    %transposed = linalg.transpose ins(%arg2 : tensor<1x8x16xf32>) outs(%0 : tensor<1x16x8xf32>) permutation = [0, 2, 1] 
    %1 = tensor.empty() : tensor<1x1x8xf32>
    %2 = linalg.fill ins(%cst : f32) outs(%1 : tensor<1x1x8xf32>) -> tensor<1x1x8xf32>
    %3 = linalg.batch_matmul ins(%extracted_slice, %transposed : tensor<1x1x16xf32>, tensor<1x16x8xf32>) outs(%2 : tensor<1x1x8xf32>) -> tensor<1x1x8xf32>
    %concat = tensor.concat dim(2) %3, %extracted_slice_0 : (tensor<1x1x8xf32>, tensor<1x1x4xf32>) -> tensor<1x1x12xf32>
    %concat_1 = tensor.concat dim(2) %arg1, %arg3 : (tensor<1x32x8xf32>, tensor<1x32x4xf32>) -> tensor<1x32x12xf32>
    %4 = tensor.empty() : tensor<1x1x32xf32>
    %5 = linalg.fill ins(%cst : f32) outs(%4 : tensor<1x1x32xf32>) -> tensor<1x1x32xf32>
    %6 = linalg.generic {indexing_maps = [#map, #map1, #map2], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%concat, %concat_1 : tensor<1x1x12xf32>, tensor<1x32x12xf32>) outs(%5 : tensor<1x1x32xf32>) attrs =  {attention.projection_placement = "key-query-rotary-shared"} {
    ^bb0(%in: f32, %in_2: f32, %out: f32):
      %7 = arith.mulf %in, %in_2 : f32
      %8 = arith.addf %out, %7 : f32
      linalg.yield %8 : f32
    } -> tensor<1x1x32xf32>
    return %6 : tensor<1x1x32xf32>
  }
}

