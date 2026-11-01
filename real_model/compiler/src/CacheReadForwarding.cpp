// SPDX-License-Identifier: MIT
// Demand-driven reads of functional cache versions. No FP reassociation.
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
using namespace mlir;
namespace {
// The original operations require valid insertion/extraction coordinates.
// This rewrite preserves that domain. Reads from the inactive alternative must
// remain conditional: arith.select(extract(update,...), extract(base,...)) would
// introduce out-of-bounds reads even when the selected result is valid.
struct ForwardRead final : OpRewritePattern<tensor::ExtractOp> {
 using OpRewritePattern::OpRewritePattern;
 LogicalResult matchAndRewrite(tensor::ExtractOp read,PatternRewriter &rw) const override {
  auto write=read.getTensor().getDefiningOp<tensor::InsertSliceOp>();
  auto gather=read->getParentOfType<linalg::GenericOp>();
  if(!write || !gather || !gather.hasPureTensorSemantics() ||
     !llvm::all_of(gather.getIteratorTypesArray(),[](utils::IteratorType t){return t==utils::IteratorType::parallel;}))return failure();
  auto src=cast<RankedTensorType>(write.getSource().getType());
  auto dst=cast<RankedTensorType>(write.getDest().getType());
  if(src.getRank()!=dst.getRank() || src.getRank()==0 || src.getRank()>5 ||
     !src.hasStaticShape() || !dst.hasStaticShape())return failure();
  for(int64_t d:src.getShape())if(d<=0)return failure();
  for(int64_t d:dst.getShape())if(d<=0)return failure();
  auto func=read->getParentOfType<func::FuncOp>();
  bool force=func && func->hasAttr("attention.force_cache_forwarding");
  // A lone cache version is already efficiently updated in-place by IREE.
  // Forward only when the base has competing observations/versions and the
  // demanded gather is much smaller than the full version being materialized.
  if(!force) {
   if(write.getDest().hasOneUse())return failure();
   long double demand=0,capacity=1;
   SmallVector<Operation *> readers;
   for(Operation *user:write.getResult().getUsers()) {
    auto extract=dyn_cast<tensor::ExtractOp>(user);
    if(!extract)return failure(); // A returned/materialized version is still needed.
    auto consumer=extract->getParentOfType<linalg::GenericOp>();
    if(!consumer || !consumer.hasPureTensorSemantics() ||
       !llvm::all_of(consumer.getIteratorTypesArray(),[](utils::IteratorType t){return t==utils::IteratorType::parallel;}))return failure();
    if(!llvm::is_contained(readers,consumer.getOperation()))readers.push_back(consumer);
   }
   for(Operation *reader:readers)for(Value result:reader->getResults()) {
    auto ty=dyn_cast<RankedTensorType>(result.getType());
    if(!ty || !ty.hasStaticShape())return failure();
    long double elements=1;for(int64_t d:ty.getShape()){if(d<=0)return failure();elements*=d;}demand+=elements;
   }
   for(int64_t d:dst.getShape())capacity*=d;
   if(capacity<2*demand)return failure();
  }
  auto offsets=write.getMixedOffsets();auto sizes=write.getMixedSizes();auto strides=write.getMixedStrides();
  // Positive static strides are required. Static offset/size bounds are
  // verified by Tensor; dynamic insertion offsets retain its original contract.
  for(OpFoldResult step:strides){auto c=getConstantIntValue(step);if(!c || *c<=0)return failure();}
  Location loc=read.getLoc();
  Value condition;SmallVector<Value> translated;
  auto materialize=[&](OpFoldResult x)->Value {
   if(auto v=dyn_cast<Value>(x))return v;
   return rw.create<arith::ConstantIndexOp>(loc,cast<IntegerAttr>(cast<Attribute>(x)).getInt());
  };
  Value zero=rw.create<arith::ConstantIndexOp>(loc,0);
  for(unsigned axis=0;axis<read.getIndices().size();++axis) {
   Value index=read.getIndices()[axis];
   auto off=getConstantIntValue(offsets[axis]);auto size=getConstantIntValue(sizes[axis]);auto step=getConstantIntValue(strides[axis]);
   if(off && size && *off==0 && *size==dst.getDimSize(axis) && *step==1){translated.push_back(index);continue;}
   Value delta=rw.create<arith::SubIOp>(loc,index,materialize(offsets[axis]));
   Value position=delta;
   Value axisCondition;
   if(*step!=1) {
    Value stride=materialize(strides[axis]);
    position=rw.create<arith::DivUIOp>(loc,delta,stride);
    Value rem=rw.create<arith::RemUIOp>(loc,delta,stride);
    axisCondition=rw.create<arith::CmpIOp>(loc,arith::CmpIPredicate::eq,rem,zero);
   }
   // Unsigned delta bounds simultaneously exclude negative differences without
   // forming offset + size*stride, avoiding that extra overflow expression.
   Value below=rw.create<arith::CmpIOp>(loc,arith::CmpIPredicate::ult,position,materialize(sizes[axis]));
   if(axisCondition)axisCondition=rw.create<arith::AndIOp>(loc,axisCondition,below);else axisCondition=below;
   // For strided regions, wrapped negative delta could divide into a small
   // quotient for an enormous stride, so retain an explicit lower-bound test.
   if(*step!=1) {
    Value above=rw.create<arith::CmpIOp>(loc,arith::CmpIPredicate::uge,index,materialize(offsets[axis]));
    axisCondition=rw.create<arith::AndIOp>(loc,axisCondition,above);
   }
   if(condition)condition=rw.create<arith::AndIOp>(loc,condition,axisCondition);else condition=axisCondition;
   translated.push_back(position);
  }
  if(!condition){rw.replaceOpWithNewOp<tensor::ExtractOp>(read,write.getSource(),translated);return success();}
  auto branch=rw.create<scf::IfOp>(loc,condition,
   [&](OpBuilder &b,Location l){Value v=b.create<tensor::ExtractOp>(l,write.getSource(),translated);b.create<scf::YieldOp>(l,v);},
   [&](OpBuilder &b,Location l){Value v=b.create<tensor::ExtractOp>(l,write.getDest(),read.getIndices());b.create<scf::YieldOp>(l,v);});
  branch->setAttr("attention.cache_read_forwarded",rw.getUnitAttr());
  rw.replaceOp(read,branch.getResults());return success();
 }
};
struct CacheReadForwarding final : PassWrapper<CacheReadForwarding,OperationPass<>> {
 MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(CacheReadForwarding)
 StringRef getArgument() const final{return "attention-forward-cache-reads";}
 StringRef getDescription() const final{return "Forward demanded gathers through competing functional cache updates";}
 void getDependentDialects(DialectRegistry &r)const override{r.insert<scf::SCFDialect,arith::ArithDialect,tensor::TensorDialect>();}
 void runOnOperation()override{
  RewritePatternSet patterns(&getContext());patterns.add<ForwardRead>(&getContext());
  if(failed(applyPatternsAndFoldGreedily(getOperation(),std::move(patterns))))signalPassFailure();
 }
};
}
void registerCacheReadForwarding(){PassRegistration<CacheReadForwarding>();}
