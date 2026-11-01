// SPDX-License-Identifier: MIT
// Fusible representation of gathered cache versions: base gather + masked patch.
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Arith/Utils/Utils.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
using namespace mlir;
namespace {
struct GatherForward final:RewritePattern {
 unsigned demandRatio;
 GatherForward(MLIRContext *ctx,unsigned ratio):RewritePattern("iree_linalg_ext.gather",1,ctx),demandRatio(ratio){}
 LogicalResult matchAndRewrite(Operation *op,PatternRewriter &rw)const override{
  if(op->getNumOperands()!=3 || op->getNumResults()!=1 || op->hasAttr("lowering_config"))return failure();
  auto write=op->getOperand(0).getDefiningOp<tensor::InsertSliceOp>();if(!write)return failure();
  auto props=dyn_cast_or_null<DictionaryAttr>(op->getPropertiesAsAttribute());
  auto dims=op->getAttrOfType<DenseI64ArrayAttr>("dimension_map");if(!dims && props)dims=props.getAs<DenseI64ArrayAttr>("dimension_map");
  if(!dims || dims.size()!=1 || dims[0]!=0)return failure();
  auto src=cast<RankedTensorType>(write.getSource().getType());auto dst=cast<RankedTensorType>(write.getDest().getType());
  auto idx=dyn_cast<RankedTensorType>(op->getOperand(1).getType());auto out=dyn_cast<RankedTensorType>(op->getResult(0).getType());
  if(!idx || !out || idx.getRank()!=1 || out.getRank()!=dst.getRank() || src.getRank()!=dst.getRank() ||
     !idx.hasStaticShape() || !src.hasStaticShape() || !dst.hasStaticShape() || !out.hasStaticShape() ||
     idx.getDimSize(0)!=out.getDimSize(0) || op->getOperand(2).getType()!=out || idx.getEncoding() ||
     dst.getRank()<1 || dst.getRank()>5)return failure();
  Type elem=idx.getElementType();auto integer=dyn_cast<IntegerType>(elem);
  if(!elem.isIndex() && (!integer || integer.getWidth()<32 || integer.getWidth()>64))return failure();
  if(integer && integer.getWidth()<64 && dst.getDimSize(0)>((int64_t(1)<<(integer.getWidth()-1))-1))return failure();
  if(src.getDimSize(0)<=0 || out.getDimSize(0)<=0 || write.getDest().hasOneUse())return failure();
  auto offsets=write.getMixedOffsets(),sizes=write.getMixedSizes(),strides=write.getMixedStrides();
  for(unsigned i=0;i<strides.size();++i){auto step=getConstantIntValue(strides[i]);if(!step || *step!=1)return failure();}
  for(unsigned i=1;i<offsets.size();++i){auto off=getConstantIntValue(offsets[i]);auto size=getConstantIntValue(sizes[i]);
   if(!off || *off!=0 || !size || *size!=dst.getDimSize(i) || src.getDimSize(i)!=dst.getDimSize(i) || out.getDimSize(i)!=dst.getDimSize(i))return failure();}
  long double demand=0,capacity=1;for(int64_t s:dst.getShape()){if(s<=0)return failure();capacity*=s;}
  for(Operation *user:write.getResult().getUsers()){
   if(user->getName().getStringRef()!="iree_linalg_ext.gather" || user->getNumOperands()<1 || user->getNumResults()!=1 || user->getOperand(0)!=write.getResult())return failure();
   auto ty=dyn_cast<RankedTensorType>(user->getResult(0).getType());if(!ty || !ty.hasStaticShape())return failure();
   long double size=1;for(int64_t d:ty.getShape()){if(d<=0)return failure();size*=d;}demand+=size;
  }
  // Avoid patch overhead when a gather already reads a substantial pool fraction.
  if(capacity<demandRatio*demand)return failure();
  Location loc=op->getLoc();Value offset=getValueOrCreateConstantIndexOp(rw,loc,offsets[0]);
  Value limit=rw.create<arith::ConstantIndexOp>(loc,src.getDimSize(0));Value zero=rw.create<arith::ConstantIndexOp>(loc,0);
  if(!elem.isIndex()){offset=rw.create<arith::IndexCastOp>(loc,elem,offset);limit=rw.create<arith::IndexCastOp>(loc,elem,limit);zero=rw.create<arith::IndexCastOp>(loc,elem,zero);}
  auto maskType=RankedTensorType::get(idx.getShape(),rw.getI1Type());
  Value ie=rw.create<tensor::EmptyOp>(loc,idx.getShape(),elem);Value me=rw.create<tensor::EmptyOp>(loc,maskType.getShape(),rw.getI1Type());
  auto identity=AffineMap::getMultiDimIdentityMap(1,rw.getContext());
  auto translate=rw.create<linalg::GenericOp>(loc,TypeRange{idx,maskType},ValueRange{op->getOperand(1)},ValueRange{ie,me},
   ArrayRef<AffineMap>{identity,identity,identity},ArrayRef<utils::IteratorType>{utils::IteratorType::parallel},
   [&](OpBuilder &b,Location l,ValueRange args){
    Value delta=b.create<arith::SubIOp>(l,args[0],offset);Value valid=b.create<arith::CmpIOp>(l,arith::CmpIPredicate::ult,delta,limit);
    Value safe=b.create<arith::SelectOp>(l,valid,delta,zero);b.create<linalg::YieldOp>(l,ValueRange{safe,valid});
   });
  auto originalProperties=op->getPropertiesAsAttribute();
  Operation *base=rw.clone(*op);base->setOperand(0,write.getDest());
  (void)base->setPropertiesFromAttribute(originalProperties,[&]{return base->emitError("invalid gather properties");});
  Operation *patch=rw.clone(*op);
  (void)patch->setPropertiesFromAttribute(originalProperties,[&]{return patch->emitError("invalid gather properties");});
  rw.modifyOpInPlace(patch,[&]{patch->setOperands(ValueRange{write.getSource(),translate.getResult(0),translate.getResult(1),base->getResult(0)});patch->setAttr("attention.cache_gather_forwarded",rw.getUnitAttr());});
  rw.replaceOp(op,patch->getResults());return success();
 }
};
struct CacheGatherForwarding final:PassWrapper<CacheGatherForwarding,OperationPass<func::FuncOp>>{
 MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(CacheGatherForwarding)
 CacheGatherForwarding()=default;
 CacheGatherForwarding(const CacheGatherForwarding &other):PassWrapper(other){}
 Option<unsigned> demandRatio{*this,"min-pool-demand-ratio",llvm::cl::desc("Minimum pool/read element ratio; >=2"),llvm::cl::init(4)};
 StringRef getArgument()const final{return "attention-forward-gather-updates";}
 StringRef getDescription()const final{return "Forward functional cache updates as base gathers and masked patches";}
 void getDependentDialects(DialectRegistry &r)const override{r.insert<arith::ArithDialect,linalg::LinalgDialect,tensor::TensorDialect>();}
 void runOnOperation()override{RewritePatternSet ps(&getContext());ps.add<GatherForward>(&getContext(),std::max(2u,unsigned(demandRatio)));if(failed(applyPatternsAndFoldGreedily(getOperation(),std::move(ps))))signalPassFailure();}
};
}
void registerCacheGatherForwarding(){PassRegistration<CacheGatherForwarding>();}
