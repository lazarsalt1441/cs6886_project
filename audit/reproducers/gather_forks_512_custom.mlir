module {
func.func @main(%base: tensor<512x16xf32>, %a: tensor<4x16xf32>, %b: tensor<4x16xf32>, %idx: tensor<8xi64>) -> tensor<16x16xf32> {
%gae = tensor.empty() : tensor<8x16xf32>
%ga = linalg.generic {indexing_maps=[affine_map<(i,j)->(i)>,affine_map<(i,j)->(i,j)>],iterator_types=["parallel","parallel"]} ins(%idx : tensor<8xi64>) outs(%gae : tensor<8x16xf32>) {
^bb0(%raw:i64,%unused:f32):
%row = arith.index_cast %raw : i64 to index
%col = linalg.index 1 : index
%start = arith.constant 128 : index
%end = arith.constant 132 : index
%ge = arith.cmpi uge, %row,%start : index
%lt = arith.cmpi ult, %row,%end : index
%inside = arith.andi %ge,%lt : i1
%x = scf.if %inside -> (f32) {
%local = arith.subi %row,%start : index
%v = tensor.extract %a[%local,%col] : tensor<4x16xf32>
scf.yield %v : f32
} else {
%v = tensor.extract %base[%row,%col] : tensor<512x16xf32>
scf.yield %v : f32
}
linalg.yield %x : f32
} -> tensor<8x16xf32>
%gbe = tensor.empty() : tensor<8x16xf32>
%gb = linalg.generic {indexing_maps=[affine_map<(i,j)->(i)>,affine_map<(i,j)->(i,j)>],iterator_types=["parallel","parallel"]} ins(%idx : tensor<8xi64>) outs(%gbe : tensor<8x16xf32>) {
^bb0(%raw:i64,%unused:f32):
%row = arith.index_cast %raw : i64 to index
%col = linalg.index 1 : index
%start = arith.constant 256 : index
%end = arith.constant 260 : index
%ge = arith.cmpi uge, %row,%start : index
%lt = arith.cmpi ult, %row,%end : index
%inside = arith.andi %ge,%lt : i1
%x = scf.if %inside -> (f32) {
%local = arith.subi %row,%start : index
%v = tensor.extract %b[%local,%col] : tensor<4x16xf32>
scf.yield %v : f32
} else {
%v = tensor.extract %base[%row,%col] : tensor<512x16xf32>
scf.yield %v : f32
}
linalg.yield %x : f32
} -> tensor<8x16xf32>
%r = tensor.concat dim(0) %ga,%gb : (tensor<8x16xf32>,tensor<8x16xf32>) -> tensor<16x16xf32>
return %r : tensor<16x16xf32>
}
}
