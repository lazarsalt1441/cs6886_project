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
%U = arith.constant 4 : index
%end = arith.addi %t,%U : index
%ge = arith.cmpi uge, %i,%t : index
%lt = arith.cmpi ult, %i,%end : index
%in = arith.andi %ge,%lt : i1
%x = scf.if %in -> (f32) {
%j = arith.subi %i,%t : index
%v = tensor.extract %b[%j] : tensor<4xf32>
scf.yield %v : f32
} else {
%earlyend = arith.addi %s, %U : index
%earlyge = arith.cmpi uge, %i, %s : index
%earlylt = arith.cmpi ult, %i, %earlyend : index
%earlyin = arith.andi %earlyge, %earlylt : i1
%early = scf.if %earlyin -> (f32) {
%j = arith.subi %i, %s : index
%v = tensor.extract %a[%j] : tensor<4xf32>
scf.yield %v : f32
} else {
%v = tensor.extract %base[%i] : tensor<64xf32>
scf.yield %v : f32
}
scf.yield %early : f32
}
%r = tensor.from_elements %x : tensor<1xf32>
return %r : tensor<1xf32>
}
}
