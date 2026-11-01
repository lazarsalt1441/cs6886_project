// Pure tensor semantics: return a modified version AND the original snapshot.
// With base=[0..63], a=[101,102], b=[201,202], output1[7:11] must be7,8,9,10.
func.func @main(%base: tensor<64xf32>, %a: tensor<2xf32>, %b: tensor<2xf32>) -> (tensor<64xf32>, tensor<64xf32>) {
  %first = tensor.insert_slice %a into %base[7] [2] [1] : tensor<2xf32> into tensor<64xf32>
  %updated = tensor.insert_slice %b into %first[9] [2] [1] : tensor<2xf32> into tensor<64xf32>
  return %updated, %base : tensor<64xf32>, tensor<64xf32>
}
