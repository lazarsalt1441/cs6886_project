module {
func.func @main(%base: tensor<4096xf32>, %patch: tensor<16xf32>, %idx: tensor<2xi64>) -> tensor<1xf32> {
%c0 = arith.constant 0 : index
%c1 = arith.constant 1 : index
%s0 = tensor.extract %idx[%c0] : tensor<2xi64>
%i0 = tensor.extract %idx[%c1] : tensor<2xi64>
%s = arith.index_cast %s0 : i64 to index
%i = arith.index_cast %i0 : i64 to index
%U = arith.constant 16 : index
%xend = arith.addi %s, %U : index
%xge = arith.cmpi uge, %i, %s : index
%xlt = arith.cmpi ult, %i, %xend : index
%xin = arith.andi %xge, %xlt : i1
%x = scf.if %xin -> (f32) {
%j = arith.subi %i, %s : index
%v = tensor.extract %patch[%j] : tensor<16xf32>
scf.yield %v : f32
} else {
%v = tensor.extract %base[%i] : tensor<4096xf32>
scf.yield %v : f32
}
%r = tensor.from_elements %x : tensor<1xf32>
return %r : tensor<1xf32>
}
}
