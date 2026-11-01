module {
func.func @main(%base: tensor<64xf32>, %patch: tensor<4xf32>, %idx: tensor<2xi64>) -> tensor<1xf32> {
%c0 = arith.constant 0 : index
%c1 = arith.constant 1 : index
%s0 = tensor.extract %idx[%c0] : tensor<2xi64>
%i0 = tensor.extract %idx[%c1] : tensor<2xi64>
%s = arith.index_cast %s0 : i64 to index
%i = arith.index_cast %i0 : i64 to index
%updated = tensor.insert_slice %patch into %base[%s] [4] [1] : tensor<4xf32> into tensor<64xf32>
%x = tensor.extract %updated[%i] : tensor<64xf32>
%r = tensor.from_elements %x : tensor<1xf32>
return %r : tensor<1xf32>
}
}
