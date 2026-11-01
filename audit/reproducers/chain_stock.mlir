module {
func.func @main(%base: tensor<64xf32>, %a: tensor<4xf32>, %b: tensor<4xf32>, %idx: tensor<3xi64>) -> tensor<1xf32> {
%c0 = arith.constant 0 : index
%c1 = arith.constant 1 : index
%c2 = arith.constant 2 : index
%a0 = tensor.extract %idx[%c0] : tensor<3xi64>
%b0 = tensor.extract %idx[%c1] : tensor<3xi64>
%i0 = tensor.extract %idx[%c2] : tensor<3xi64>
%s = arith.index_cast %a0 : i64 to index
%t = arith.index_cast %b0 : i64 to index
%i = arith.index_cast %i0 : i64 to index
%u = tensor.insert_slice %a into %base[%s] [4] [1] : tensor<4xf32> into tensor<64xf32>
%v = tensor.insert_slice %b into %u[%t] [4] [1] : tensor<4xf32> into tensor<64xf32>
%x = tensor.extract %v[%i] : tensor<64xf32>
%r = tensor.from_elements %x : tensor<1xf32>
return %r : tensor<1xf32>
}
}
