module {
func.func @main(%q: tensor<1x1x20xf32>,%l: tensor<1x32x8xf32>,%w: tensor<1x8x16xf32>,%kr: tensor<1x32x4xf32>) -> tensor<1x1x32xf32> attributes {attention.allow_reassociation} {
%zero = arith.constant 0.0 : f32
%contentempty = tensor.empty() : tensor<1x32x16xf32>
%contentinit = linalg.fill ins(%zero : f32) outs(%contentempty : tensor<1x32x16xf32>) -> tensor<1x32x16xf32>
%content = linalg.batch_matmul ins(%l,%w : tensor<1x32x8xf32>,tensor<1x8x16xf32>) outs(%contentinit : tensor<1x32x16xf32>) -> tensor<1x32x16xf32>
%key = tensor.concat dim(2) %content,%kr : (tensor<1x32x16xf32>,tensor<1x32x4xf32>) -> tensor<1x32x20xf32>
%te = tensor.empty() : tensor<1x20x32xf32>
%kt = linalg.transpose ins(%key : tensor<1x32x20xf32>) outs(%te : tensor<1x20x32xf32>) permutation = [0,2,1]
%rempty = tensor.empty() : tensor<1x1x32xf32>
%rinit = linalg.fill ins(%zero : f32) outs(%rempty : tensor<1x1x32xf32>) -> tensor<1x1x32xf32>
%r = linalg.batch_matmul ins(%q,%kt : tensor<1x1x20xf32>,tensor<1x20x32xf32>) outs(%rinit : tensor<1x1x32xf32>) -> tensor<1x1x32xf32>
return %r : tensor<1x1x32xf32>
}
}