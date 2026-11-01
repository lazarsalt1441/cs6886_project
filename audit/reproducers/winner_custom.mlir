module {
func.func @main(%base: tensor<4096xf32>, %a: tensor<16xf32>, %b: tensor<16xf32>, %cond: tensor<1xi32>) -> tensor<4096xf32> {
%c0 = arith.constant 0 : index
%c = tensor.extract %cond[%c0] : tensor<1xi32>
%z = arith.constant 0 : i32
%yes = arith.cmpi ne, %c, %z : i32
%p = arith.select %yes, %a, %b : tensor<16xf32>
%r = tensor.insert_slice %p into %base[128] [16] [1] : tensor<16xf32> into tensor<4096xf32>
return %r : tensor<4096xf32>
}
}
