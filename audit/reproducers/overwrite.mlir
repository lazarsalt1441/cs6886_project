module {
func.func @main(%base: tensor<64xf32>, %a: tensor<4xf32>, %b: tensor<4xf32>) -> tensor<64xf32> {
%u = tensor.insert_slice %a into %base[8] [4] [1] : tensor<4xf32> into tensor<64xf32>
%r = tensor.insert_slice %b into %u[8] [4] [1] : tensor<4xf32> into tensor<64xf32>
return %r : tensor<64xf32>
}
}
