// SPDX-License-Identifier: MIT
// Normalize optional unit index-component axes before gather vectorization.
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
using namespace mlir;
namespace {
struct GatherIndexProjection final : PassWrapper<GatherIndexProjection,OperationPass<ModuleOp>> {
 MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(GatherIndexProjection)
 StringRef getArgument()const final{return "attention-canonicalize-gather-indices";}
 StringRef getDescription()const final{return "Normalize depth-one gather index component axes to preserve batch projection";}
 void getDependentDialects(DialectRegistry &r)const override{r.insert<tensor::TensorDialect>();}
 void runOnOperation()override {
  getOperation().walk([&](Operation *op){
   if(op->getName().getStringRef()!="iree_linalg_ext.gather" ||
      (op->getNumOperands()!=3 && op->getNumOperands()!=4) || op->getNumResults()!=1)return;
   auto props=dyn_cast_or_null<DictionaryAttr>(op->getPropertiesAsAttribute());
   auto dims=op->getAttrOfType<DenseI64ArrayAttr>("dimension_map");
   if(!dims && props)dims=props.getAs<DenseI64ArrayAttr>("dimension_map");
   if(!dims || dims.size()!=1 || dims[0]!=0)return;
   auto source=dyn_cast<RankedTensorType>(op->getOperand(0).getType());
   auto indices=dyn_cast<RankedTensorType>(op->getOperand(1).getType());
   auto output=dyn_cast<RankedTensorType>(op->getOperand(op->getNumOperands()-1).getType());
   if(!source || !indices || !output || indices.getEncoding() ||
      source.getRank()<1 || indices.getRank()<2 ||
      !(indices.getElementType().isIntOrIndex()) ||
      op->getResult(0).getType()!=output || source.getElementType()!=output.getElementType())return;
   int64_t batchRank=output.getRank()-(source.getRank()-1);
   if(batchRank<1 || indices.getRank()!=batchRank+1 || indices.getDimSize(batchRank)!=1)return;
   for(int64_t i=0;i<batchRank;++i)
    if(!ShapedType::isDynamic(indices.getDimSize(i)) && !ShapedType::isDynamic(output.getDimSize(i)) &&
       indices.getDimSize(i)!=output.getDimSize(i))return;
   SmallVector<ReassociationIndices> groups;
   for(int64_t i=0;i<batchRank;++i)groups.push_back({i});
   groups.back().push_back(batchRank);
   auto reduced=RankedTensorType::get(indices.getShape().take_front(batchRank),indices.getElementType());
   OpBuilder builder(op);
   Value collapsed=builder.create<tensor::CollapseShapeOp>(op->getLoc(),reduced,op->getOperand(1),groups);
   op->setOperand(1,collapsed);op->setAttr("attention.index_projection_normalized",builder.getUnitAttr());
  });
 }
};
}
void registerGatherIndexProjection(){PassRegistration<GatherIndexProjection>();}
