// SPDX-License-Identifier: MIT
// Commit a selected functional cache version by selecting only its update.
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
using namespace mlir;
namespace {
bool equalMixed(ArrayRef<OpFoldResult> a,ArrayRef<OpFoldResult> b){
 if(a.size()!=b.size())return false;
 for(auto [x,y]:llvm::zip(a,b)){
  auto cx=getConstantIntValue(x),cy=getConstantIntValue(y);
  if(cx && cy){if(*cx!=*cy)return false;}else if(x!=y)return false;
 }
 return true;
}
struct FactorCommit final:OpRewritePattern<linalg::GenericOp>{
 using OpRewritePattern::OpRewritePattern;
 LogicalResult matchAndRewrite(linalg::GenericOp op,PatternRewriter &rw)const override{
  if(op.getNumDpsInputs()!=3 || op.getNumDpsInits()!=1 || op.getNumResults()!=1 || op->hasAttr("lowering_config"))return failure();
  auto ty=dyn_cast<RankedTensorType>(op.getResult(0).getType());auto cond=dyn_cast<RankedTensorType>(op.getDpsInputs()[0].getType());
  if(!ty || !ty.hasStaticShape() || ty.getEncoding() || ty.getRank()<1 || ty.getRank()>5 || !cond || cond.getRank()!=0 || !cond.getElementType().isInteger(1) || op.getNumLoops()!=ty.getRank() || !llvm::all_of(op.getIteratorTypesArray(),[](utils::IteratorType t){return t==utils::IteratorType::parallel;}))return failure();
  auto maps=op.getIndexingMapsArray();if(maps.size()!=4 || maps[0].getNumResults()!=0 || !maps[1].isIdentity() || !maps[2].isIdentity() || !maps[3].isIdentity())return failure();
  Block &body=op.getRegion().front();if(body.getOperations().size()!=2)return failure();
  auto select=dyn_cast<arith::SelectOp>(body.front());auto yield=dyn_cast<linalg::YieldOp>(body.back());
  if(!select || !yield || yield.getNumOperands()!=1 || yield.getOperand(0)!=select.getResult() || select.getCondition()!=body.getArgument(0) || select.getTrueValue()!=body.getArgument(1) || select.getFalseValue()!=body.getArgument(2))return failure();
  auto a=op.getDpsInputs()[1].getDefiningOp<tensor::InsertSliceOp>(),b=op.getDpsInputs()[2].getDefiningOp<tensor::InsertSliceOp>();
  if(!a || !b || a.getDest()!=b.getDest() || a.getType()!=ty || b.getType()!=ty || a.getSource().getType()!=b.getSource().getType())return failure();
  auto update=dyn_cast<RankedTensorType>(a.getSource().getType());if(!update || !update.hasStaticShape() || update.getEncoding() || update.getRank()!=ty.getRank())return failure();
  for(int64_t s:update.getShape())if(s<=0)return failure();
  if(!equalMixed(a.getMixedOffsets(),b.getMixedOffsets()) || !equalMixed(a.getMixedSizes(),b.getMixedSizes()) || !equalMixed(a.getMixedStrides(),b.getMixedStrides()))return failure();
  for(OpFoldResult stride:a.getMixedStrides()){auto v=getConstantIntValue(stride);if(!v || *v<=0)return failure();}
  Location loc=op.getLoc();Value empty=rw.create<tensor::EmptyOp>(loc,update.getShape(),update.getElementType());
  auto selected=rw.create<linalg::GenericOp>(loc,TypeRange{update},ValueRange{op.getDpsInputs()[0],a.getSource(),b.getSource()},ValueRange{empty},maps,op.getIteratorTypesArray(),
   [&](OpBuilder &builder,Location l,ValueRange args){Value value=builder.create<arith::SelectOp>(l,args[0],args[1],args[2]);builder.create<linalg::YieldOp>(l,value);});
  auto committed=rw.create<tensor::InsertSliceOp>(loc,selected.getResult(0),a.getDest(),a.getMixedOffsets(),a.getMixedSizes(),a.getMixedStrides());committed->setAttr("attention.commit_factored",rw.getUnitAttr());rw.replaceOp(op,committed.getResult());return success();
 }
};
struct CacheCommitFactoring final:PassWrapper<CacheCommitFactoring,OperationPass<func::FuncOp>>{
 MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(CacheCommitFactoring)
 StringRef getArgument()const final{return "attention-factor-cache-commit";}
 StringRef getDescription()const final{return "Factor invariant selection of matching cache updates before read forwarding";}
 void getDependentDialects(DialectRegistry &r)const override{r.insert<arith::ArithDialect,linalg::LinalgDialect,tensor::TensorDialect>();}
 void runOnOperation()override{RewritePatternSet ps(&getContext());ps.add<FactorCommit>(&getContext());if(failed(applyPatternsAndFoldGreedily(getOperation(),std::move(ps))))signalPassFailure();}
};
}
void registerCacheCommitFactoring(){PassRegistration<CacheCommitFactoring>();}
