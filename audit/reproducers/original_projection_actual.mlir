module {
  func.func @main(%arg0: tensor<1x1x16xf32>, %arg1: tensor<1x32x8xf32>, %arg2: tensor<1x8x16xf32>) -> tensor<1x1x32xf32> attributes {attention.allow_reassociation} {
    %cst = arith.constant 0.000000e+00 : f32
    %0 = tensor.empty() : tensor<1x16x8xf32>
    %transposed = linalg.transpose ins(%arg2 : tensor<1x8x16xf32>) outs(%0 : tensor<1x16x8xf32>) permutation = [0, 2, 1] 
    %1 = tensor.empty() : tensor<1x1x8xf32>
    %2 = linalg.fill ins(%cst : f32) outs(%1 : tensor<1x1x8xf32>) -> tensor<1x1x8xf32>
    %3 = linalg.batch_matmul ins(%arg0, %transposed : tensor<1x1x16xf32>, tensor<1x16x8xf32>) outs(%2 : tensor<1x1x8xf32>) -> tensor<1x1x8xf32>
    %4 = tensor.empty() : tensor<1x8x32xf32>
    %transposed_0 = linalg.transpose ins(%arg1 : tensor<1x32x8xf32>) outs(%4 : tensor<1x8x32xf32>) permutation = [0, 2, 1] 
    %5 = tensor.empty() : tensor<1x1x32xf32>
    %6 = linalg.fill ins(%cst : f32) outs(%5 : tensor<1x1x32xf32>) -> tensor<1x1x32xf32>
    %7 = linalg.batch_matmul {attention.projection_placement = "key-query"} ins(%3, %transposed_0 : tensor<1x1x8xf32>, tensor<1x8x32xf32>) outs(%6 : tensor<1x1x32xf32>) -> tensor<1x1x32xf32>
    return %7 : tensor<1x1x32xf32>
  }
}

