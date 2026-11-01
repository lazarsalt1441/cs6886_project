#map = affine_map<(d0, d1) -> (d0)>
#map1 = affine_map<(d0, d1) -> (d0, d1)>
module {
  func.func @main(%arg0: tensor<512x16xf32>, %arg1: tensor<4x16xf32>, %arg2: tensor<4x16xf32>, %arg3: tensor<8xi64>) -> tensor<16x16xf32> {
    %c256 = arith.constant 256 : index
    %c4 = arith.constant 4 : index
    %c128 = arith.constant 128 : index
    %0 = tensor.empty() : tensor<8x16xf32>
    %1 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "parallel"]} ins(%arg3 : tensor<8xi64>) outs(%0 : tensor<8x16xf32>) {
    ^bb0(%in: i64, %out: f32):
      %4 = arith.index_cast %in : i64 to index
      %5 = linalg.index 1 : index
      %6 = arith.subi %4, %c128 : index
      %7 = arith.cmpi ult, %6, %c4 : index
      %8 = scf.if %7 -> (f32) {
        %extracted = tensor.extract %arg1[%6, %5] : tensor<4x16xf32>
        scf.yield %extracted : f32
      } else {
        %extracted = tensor.extract %arg0[%4, %5] : tensor<512x16xf32>
        scf.yield %extracted : f32
      } {attention.cache_read_forwarded}
      linalg.yield %8 : f32
    } -> tensor<8x16xf32>
    %2 = tensor.empty() : tensor<8x16xf32>
    %3 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "parallel"]} ins(%arg3 : tensor<8xi64>) outs(%2 : tensor<8x16xf32>) {
    ^bb0(%in: i64, %out: f32):
      %4 = arith.index_cast %in : i64 to index
      %5 = linalg.index 1 : index
      %6 = arith.subi %4, %c256 : index
      %7 = arith.cmpi ult, %6, %c4 : index
      %8 = scf.if %7 -> (f32) {
        %extracted = tensor.extract %arg2[%6, %5] : tensor<4x16xf32>
        scf.yield %extracted : f32
      } else {
        %extracted = tensor.extract %arg0[%4, %5] : tensor<512x16xf32>
        scf.yield %extracted : f32
      } {attention.cache_read_forwarded}
      linalg.yield %8 : f32
    } -> tensor<8x16xf32>
    %concat = tensor.concat dim(0) %1, %3 : (tensor<8x16xf32>, tensor<8x16xf32>) -> tensor<16x16xf32>
    return %concat : tensor<16x16xf32>
  }
}

