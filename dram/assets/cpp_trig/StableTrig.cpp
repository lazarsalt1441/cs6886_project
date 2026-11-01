// SPDX-License-Identifier: MIT
// Guarded f64 argument reduction for small f32 tensor sin/cos kernels.
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
using namespace mlir;
namespace {
template<typename Op> struct Stable final:OpRewritePattern<Op>{
 using OpRewritePattern<Op>::OpRewritePattern;
 LogicalResult matchAndRewrite(Op op,PatternRewriter &rw)const override{
  if(!op.getOperand().getType().isF32() || op->hasAttr("attention.trig_reduced"))return failure();
  auto parent=op->template getParentOfType<linalg::GenericOp>();
  if(!parent || parent.getNumResults()!=1 || parent->hasAttr("lowering_config"))return failure();
  auto t=dyn_cast<RankedTensorType>(parent.getResult(0).getType());
  if(!t || !t.hasStaticShape() || t.getEncoding() || t.getNumElements()>4096 || t.getNumElements()<1)return failure();
  Location l=op.getLoc();Value x=op.getOperand();
  auto cf=[&](double v){return rw.create<arith::ConstantOp>(l,rw.getF64FloatAttr(v));};
  Value xd=rw.create<arith::ExtFOp>(l,rw.getF64Type(),x);
  Value turns=rw.create<arith::MulFOp>(l,xd,cf(0.15915494309189533577));
  turns=rw.create<arith::AddFOp>(l,turns,cf(0.5));
  Value n=rw.create<math::FloorOp>(l,turns);
  Value shift=rw.create<arith::MulFOp>(l,n,cf(6.28318530717958647693));
  Value reduced=rw.create<arith::SubFOp>(l,xd,shift);
  reduced=rw.create<arith::TruncFOp>(l,rw.getF32Type(),reduced);
  auto repaired=rw.create<Op>(l,reduced);repaired->setAttr("attention.trig_reduced",rw.getUnitAttr());
  auto original=rw.create<Op>(l,x);original->setAttr("attention.trig_reduced",rw.getUnitAttr());
  Value ax=rw.create<math::AbsFOp>(l,x);
  Value lo=rw.create<arith::ConstantOp>(l,rw.getF32FloatAttr(32.0));
  Value hi=rw.create<arith::ConstantOp>(l,rw.getF32FloatAttr(8192.0));
  Value gt=rw.create<arith::CmpFOp>(l,arith::CmpFPredicate::OGT,ax,lo);
  Value le=rw.create<arith::CmpFOp>(l,arith::CmpFPredicate::OLE,ax,hi);
  Value use=rw.create<arith::AndIOp>(l,gt,le);
  auto result=rw.create<arith::SelectOp>(l,use,repaired.getResult(),original.getResult());
  result->setAttr("attention.stable_trig",rw.getUnitAttr());rw.replaceOp(op,result.getResult());return success();
 }
};
struct StableTrig final:PassWrapper<StableTrig,OperationPass<>>{
 MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(StableTrig)
 StringRef getArgument()const final{return "attention-stable-trig";}
 StringRef getDescription()const final{return "Guarded accurate argument reduction in small f32 sin/cos kernels";}
 void getDependentDialects(DialectRegistry&r)const final{r.insert<arith::ArithDialect,math::MathDialect,linalg::LinalgDialect>();}
 void runOnOperation()final{RewritePatternSet p(&getContext());p.add<Stable<math::SinOp>,Stable<math::CosOp>>(&getContext());if(failed(applyPatternsAndFoldGreedily(getOperation(),std::move(p))))signalPassFailure();}
};
}
void registerStableTrig(){PassRegistration<StableTrig>();}
