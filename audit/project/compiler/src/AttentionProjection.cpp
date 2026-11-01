// SPDX-License-Identifier: MIT
// Cost-directed reassociation of batched linear projections around attention.
// Floating-point reassociation is opt-in: it is not an IEEE-exact rewrite.
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include <optional>
#include <cmath>
#include "AttentionBounds.h"
using namespace mlir;
namespace {
struct Matrix {
  Value value;
  RankedTensorType type;
};
std::optional<Matrix> matrix(Value value) {
  auto t=dyn_cast<RankedTensorType>(value.getType());
  if(!t || t.getRank()!=3 || !t.hasStaticShape() || !t.getElementType().isF32())
    return std::nullopt;
  for(auto s:t.getShape()) if(s<=0) return std::nullopt;
  return Matrix{value,t};
}
bool zeroInit(linalg::BatchMatmulOp op) {
  if(!op || !op.hasPureTensorSemantics()) return false;
  Block &body=op.getRegion().front();
  if(body.getOperations().size()!=3) return false;
  auto mul=dyn_cast<arith::MulFOp>(body.front());
  auto add=dyn_cast<arith::AddFOp>(*std::next(body.begin()));
  auto yield=dyn_cast<linalg::YieldOp>(body.getTerminator());
  if(!mul || !add || !yield || mul.getLhs()!=body.getArgument(0) ||
     mul.getRhs()!=body.getArgument(1) || add.getLhs()!=body.getArgument(2) ||
     add.getRhs()!=mul.getResult() || yield.getOperand(0)!=add.getResult()) return false;
  auto fill=op.getDpsInits()[0].getDefiningOp<linalg::FillOp>();
  if(!fill) return false;
  Attribute val;
  if(!matchPattern(fill.getInputs()[0],m_Constant(&val))) return false;
  auto f=dyn_cast<FloatAttr>(val);
  return f && f.getValue().isZero();
}
Value transpose(PatternRewriter &rw,Location loc,Value value) {
  auto t=cast<RankedTensorType>(value.getType());
  auto out=RankedTensorType::get({t.getDimSize(0),t.getDimSize(2),t.getDimSize(1)},t.getElementType());
  Value empty=rw.create<tensor::EmptyOp>(loc,out.getShape(),out.getElementType());
  return rw.create<linalg::TransposeOp>(loc,value,empty,ArrayRef<int64_t>{0,2,1}).getResult()[0];
}
Value multiply(PatternRewriter &rw,Location loc,Value a,Value b) {
  auto at=cast<RankedTensorType>(a.getType());auto bt=cast<RankedTensorType>(b.getType());
  auto t=RankedTensorType::get({at.getDimSize(0),at.getDimSize(1),bt.getDimSize(2)},at.getElementType());
  Value empty=rw.create<tensor::EmptyOp>(loc,t.getShape(),t.getElementType());
  Value zero=rw.create<arith::ConstantOp>(loc,rw.getF32FloatAttr(0.0));
  Value init=rw.create<linalg::FillOp>(loc,ValueRange{zero},ValueRange{empty}).getResult(0);
  auto out=rw.create<linalg::BatchMatmulOp>(loc,TypeRange{t},ValueRange{a,b},ValueRange{init});
  return out.getResult(0);
}
Value sliceColumns(PatternRewriter &rw,Location loc,Value value,int64_t start,int64_t width) {
  auto type=cast<RankedTensorType>(value.getType());
  SmallVector<OpFoldResult> offsets={rw.getIndexAttr(0),rw.getIndexAttr(0),rw.getIndexAttr(start)};
  SmallVector<OpFoldResult> sizes={rw.getIndexAttr(type.getDimSize(0)),rw.getIndexAttr(type.getDimSize(1)),rw.getIndexAttr(width)};
  SmallVector<OpFoldResult> strides(3,rw.getIndexAttr(1));
  return rw.create<tensor::ExtractSliceOp>(loc,value,offsets,sizes,strides);
}
// Recover a head-shared source only for a proven pure broadcast copy.
Value sharedSource(Value value) {
  auto g=value.getDefiningOp<linalg::GenericOp>();
  if(!g || g.getNumDpsInputs()!=1 || g.getNumDpsInits()!=1 || g.getNumResults()!=1)
    return value;
  Value original=g.getDpsInputs()[0];
  auto input=matrix(original);auto output=matrix(value);
  auto maps=g.getIndexingMapsArray();auto ctx=g.getContext();
  AffineMap expected;
  if(!input) {
    auto collapsed=original.getDefiningOp<tensor::CollapseShapeOp>();
    if(!collapsed) return value;
    auto groups=collapsed.getReassociationIndices();
    if(groups!=SmallVector<ReassociationIndices>{{0,1},{2}}) return value;
    original=collapsed.getSrc();input=matrix(original);
    expected=AffineMap::get(3,0,{getAffineDimExpr(1,ctx),getAffineDimExpr(2,ctx)},ctx);
  } else {
    expected=AffineMap::get(3,0,{getAffineConstantExpr(0,ctx),getAffineDimExpr(1,ctx),getAffineDimExpr(2,ctx)},ctx);
  }
  if(!input || !output || input->type.getDimSize(0)!=1 ||
     input->type.getDimSize(1)!=output->type.getDimSize(1) ||
     input->type.getDimSize(2)!=output->type.getDimSize(2)) return value;
  if(maps[0]!=expected || !maps[1].isIdentity()) return value;
  for(auto it:g.getIteratorTypesArray()) if(it!=utils::IteratorType::parallel) return value;
  Block &body=g.getRegion().front();auto yield=dyn_cast<linalg::YieldOp>(body.getTerminator());
  if(!yield || body.getOperations().size()!=1 || yield.getOperand(0)!=body.getArgument(0)) return value;
  return input->value;
}
Value sharedConcat(PatternRewriter &rw,Location loc,Value a,Value b) {
  Value sa=sharedSource(a),sb=sharedSource(b);
  auto at=cast<RankedTensorType>(sa.getType()),bt=cast<RankedTensorType>(sb.getType());
  if(at.getDimSize(0)==bt.getDimSize(0)) return rw.create<tensor::ConcatOp>(loc,2,ValueRange{sa,sb});
  return rw.create<tensor::ConcatOp>(loc,2,ValueRange{a,b});
}
Value scoreContract(PatternRewriter &rw,Location loc,Value q,Value k) {
  auto qt=cast<RankedTensorType>(q.getType()),kt=cast<RankedTensorType>(k.getType());
  auto result=RankedTensorType::get({qt.getDimSize(0),qt.getDimSize(1),kt.getDimSize(1)},qt.getElementType());
  Value empty=rw.create<tensor::EmptyOp>(loc,result.getShape(),result.getElementType());
  Value zero=rw.create<arith::ConstantOp>(loc,rw.getF32FloatAttr(0.0));
  Value init=rw.create<linalg::FillOp>(loc,ValueRange{zero},ValueRange{empty}).getResult(0);
  auto d=[&](unsigned i){return getAffineDimExpr(i,rw.getContext());};
  auto head=kt.getDimSize(0)==1?getAffineConstantExpr(0,rw.getContext()):d(0);
  SmallVector<AffineMap> maps={AffineMap::get(4,0,{d(0),d(1),d(3)},rw.getContext()),
    AffineMap::get(4,0,{head,d(2),d(3)},rw.getContext()),AffineMap::get(4,0,{d(0),d(1),d(2)},rw.getContext())};
  auto out=rw.create<linalg::GenericOp>(loc,TypeRange{result},ValueRange{q,k},ValueRange{init},maps,
    ArrayRef<utils::IteratorType>{utils::IteratorType::parallel,utils::IteratorType::parallel,utils::IteratorType::parallel,utils::IteratorType::reduction},
    [&](OpBuilder &b,Location here,ValueRange args){
      Value product=b.create<arith::MulFOp>(here,args[0],args[1]);
      Value sum=b.create<arith::AddFOp>(here,args[2],product);b.create<linalg::YieldOp>(here,sum);
    });
  return out.getResult(0);
}
// Cost in scalar multiply-accumulates, represented in long double to avoid
// overflow during profitability comparisons on adversarial static dimensions.
long double mac(int64_t batch,int64_t m,int64_t n,int64_t k) {
  return static_cast<long double>(batch)*m*n*k;
}
struct ProjectionPattern : OpRewritePattern<linalg::BatchMatmulOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(linalg::BatchMatmulOp outer,PatternRewriter &rw) const override {
    auto func=outer->getParentOfType<func::FuncOp>();
    if(!func || !func->hasAttr("attention.allow_reassociation") || outer->hasAttr("attention.projection_placement") || !zeroInit(outer)) return failure();
    auto a=matrix(outer.getDpsInputs()[0]);auto b=matrix(outer.getDpsInputs()[1]);
    if(!a || !b) return failure();
    auto loc=outer.getLoc();int64_t batch=a->type.getDimSize(0);
    if(batch!=b->type.getDimSize(0)) return failure();
    // Q (C W)^T -> (Q W^T) C^T. Common in unabsorbed latent-KV attention.
    auto trans=b->value.getDefiningOp<linalg::TransposeOp>();
    if(!func->hasAttr("attention.disable_key_reassociation") && trans && trans.getPermutation()==ArrayRef<int64_t>({0,2,1}) && trans->hasOneUse()) {
      auto cat=trans.getInput().getDefiningOp<tensor::ConcatOp>();
      if(cat && cat.getDim()==2 && cat.getInputs().size()==2 && cat->hasOneUse()) {
        auto projected=cat.getInputs()[0].getDefiningOp<linalg::BatchMatmulOp>();
        auto rope=matrix(cat.getInputs()[1]);
        if(projected && zeroInit(projected) && projected->hasOneUse() && rope) {
          auto c=matrix(projected.getDpsInputs()[0]),w=matrix(projected.getDpsInputs()[1]);
          if(c && w && c->type.getDimSize(0)==batch && w->type.getDimSize(0)==batch) {
            int64_t m=a->type.getDimSize(1),n=c->type.getDimSize(1),l=c->type.getDimSize(2),d=w->type.getDimSize(2),r=rope->type.getDimSize(2);
            if(rope->type.getDimSize(0)!=batch || rope->type.getDimSize(1)!=n ||
               a->type.getDimSize(2)!=d+r) return failure();
            long double before=mac(batch,n,d,l)+mac(batch,m,n,d+r);
            long double after=mac(batch,m,l,d)+mac(batch,m,n,l+r);
            if((func->hasAttr("attention.force_reassociation") || after*1.05L<before) &&
               attention_bounds::safeTriple(func,a->value,w->value,c->value,d,l)) {
              Value qc=sliceColumns(rw,loc,a->value,0,d),qr=sliceColumns(rw,loc,a->value,d,r);
              Value absorbed=multiply(rw,loc,qc,transpose(rw,loc,w->value));
              Value joinedQ=rw.create<tensor::ConcatOp>(loc,2,ValueRange{absorbed,qr});
              Value joinedK=func->hasAttr("attention.disable_shared_concat") ?
                rw.create<tensor::ConcatOp>(loc,2,ValueRange{c->value,rope->value}).getResult() :
                sharedConcat(rw,loc,c->value,rope->value);
              Value result=scoreContract(rw,loc,joinedQ,joinedK);
              result.getDefiningOp()->setAttr("attention.projection_placement",rw.getStringAttr("key-query-rotary-shared"));
              rw.replaceOp(outer,result);return success();
            }
          }
        }
      }
      auto inner=trans.getInput().getDefiningOp<linalg::BatchMatmulOp>();
      if(inner && zeroInit(inner) && inner->hasOneUse()) {
        auto c=matrix(inner.getDpsInputs()[0]);auto w=matrix(inner.getDpsInputs()[1]);
        if(c && w && c->type.getDimSize(0)==batch && w->type.getDimSize(0)==batch) {
          auto m=a->type.getDimSize(1),n=c->type.getDimSize(1),d=a->type.getDimSize(2),l=c->type.getDimSize(2);
          long double before=mac(batch,n,d,l)+mac(batch,m,n,d);
          long double after=mac(batch,m,l,d)+mac(batch,m,n,l);
          if((func->hasAttr("attention.force_reassociation") || after*1.05L<before) &&
             attention_bounds::safeTriple(func,a->value,w->value,c->value,d,l)) {
            Value qw=multiply(rw,loc,a->value,transpose(rw,loc,w->value));
            Value result=multiply(rw,loc,qw,transpose(rw,loc,c->value));
            result.getDefiningOp()->setAttr("attention.projection_placement",rw.getStringAttr("key-query"));
            rw.replaceOp(outer,result);return success();
          }
        }
      }
    }
    // A (B C) -> (A B) C. Moves value projection after attention aggregation.
    auto right=b->value.getDefiningOp<linalg::BatchMatmulOp>();
    if(!func->hasAttr("attention.disable_value_reassociation") && right && zeroInit(right) && right->hasOneUse()) {
      auto x=matrix(right.getDpsInputs()[0]);auto y=matrix(right.getDpsInputs()[1]);
      if(x && y && x->type.getDimSize(0)==batch && y->type.getDimSize(0)==batch) {
        auto m=a->type.getDimSize(1),n=a->type.getDimSize(2),k=x->type.getDimSize(2),p=y->type.getDimSize(2);
        long double before=mac(batch,n,p,k)+mac(batch,m,p,n);
        long double after=mac(batch,m,k,n)+mac(batch,m,p,k);
        if((func->hasAttr("attention.force_reassociation") || after*1.05L<before) &&
           attention_bounds::safeTriple(func,a->value,x->value,y->value,n,k)) {
          Value ab=multiply(rw,loc,a->value,x->value);
          Value result=multiply(rw,loc,ab,y->value);
          result.getDefiningOp()->setAttr("attention.projection_placement",rw.getStringAttr("value-output"));
          rw.replaceOp(outer,result);return success();
        }
      }
    }
    // (A B) C -> A (B C), included for prefill and general chain placement.
    auto left=a->value.getDefiningOp<linalg::BatchMatmulOp>();
    if(left && zeroInit(left) && left->hasOneUse()) {
      auto x=matrix(left.getDpsInputs()[0]);auto y=matrix(left.getDpsInputs()[1]);
      if(x && y && x->type.getDimSize(0)==batch && y->type.getDimSize(0)==batch) {
        auto m=x->type.getDimSize(1),k=x->type.getDimSize(2),n=y->type.getDimSize(2),p=b->type.getDimSize(2);
        long double before=mac(batch,m,n,k)+mac(batch,m,p,n);
        long double after=mac(batch,k,p,n)+mac(batch,m,p,k);
        if(after*1.05L<before && attention_bounds::safeTriple(func,x->value,y->value,b->value,k,n)) {
          Value bc=multiply(rw,loc,y->value,b->value);
          Value result=multiply(rw,loc,x->value,bc);
          result.getDefiningOp()->setAttr("attention.projection_placement",rw.getStringAttr("right-associated"));
          rw.replaceOp(outer,result);return success();
        }
      }
    }
    return failure();
  }
};
// The external Torch-MLIR attention operation is consumed in generic textual
// form. No private IREE C++ ABI is required; its schema is verified again by
// the pinned IREE compiler after this standalone transformation.
struct NativeProjectionPattern : RewritePattern {
  NativeProjectionPattern(MLIRContext *ctx) : RewritePattern("tm_tensor.attention",1,ctx) {}
  LogicalResult matchAndRewrite(Operation *op,PatternRewriter &rw) const override {
    auto func=op->getParentOfType<func::FuncOp>();
    if(!func || !func->hasAttr("attention.allow_reassociation") ||
       op->getNumOperands()!=4 || op->getNumResults()!=1 || op->getNumRegions()!=0)
      return failure();
    if(!func->hasAttr("attention.enable_native_projection")) return failure();
    auto originalProperties=dyn_cast_or_null<DictionaryAttr>(op->getPropertiesAsAttribute());
    if(!originalProperties) return failure();
    auto explicitScale=originalProperties.getAs<FloatAttr>("scale");
    auto q=matrix(op->getOperand(0));
    Value keyValue=op->getOperand(1),rotaryKey;
    auto concat=keyValue.getDefiningOp<tensor::ConcatOp>();
    if(concat && concat.getDim()==2 && concat.getInputs().size()==2 && concat->hasOneUse()) {
      keyValue=concat.getInputs()[0];rotaryKey=concat.getInputs()[1];
    }
    auto key=keyValue.getDefiningOp<linalg::BatchMatmulOp>();
    auto value=op->getOperand(2).getDefiningOp<linalg::BatchMatmulOp>();
    if(!q || !zeroInit(key) || !zeroInit(value) || !key->hasOneUse() || !value->hasOneUse())
      return failure();
    auto c=matrix(key.getDpsInputs()[0]);auto wk=matrix(key.getDpsInputs()[1]);
    auto cv=matrix(value.getDpsInputs()[0]);auto wv=matrix(value.getDpsInputs()[1]);
    auto init=op->getOperand(3).getDefiningOp<linalg::FillOp>();
    Attribute initial;
    if(!c || !wk || !cv || !wv || c->value!=cv->value || !init ||
       !matchPattern(init.getInputs()[0],m_Constant(&initial))) return failure();
    auto initialFloat=dyn_cast<FloatAttr>(initial);
    if(!initialFloat || !initialFloat.getValue().isZero()) return failure();
    int64_t h=q->type.getDimSize(0),m=q->type.getDimSize(1),d=q->type.getDimSize(2);
    int64_t n=c->type.getDimSize(1),l=c->type.getDimSize(2),dv=wv->type.getDimSize(2);
    int64_t originalWidth=d,rotaryWidth=0;
    if(rotaryKey) {
      auto rt=matrix(rotaryKey);
      if(!rt || rt->type.getDimSize(0)!=h || rt->type.getDimSize(1)!=n) return failure();
      rotaryWidth=rt->type.getDimSize(2);d-=rotaryWidth;
      if(d<=0) return failure();
    }
    if(c->type.getDimSize(0)!=h || wk->type.getDimSize(0)!=h ||
       wv->type.getDimSize(0)!=h || wk->type.getDimSize(1)!=l ||
       wv->type.getDimSize(1)!=l || wk->type.getDimSize(2)!=d) return failure();
    if(func->hasAttr("attention.require_range_safety")) return failure();
    bool force=func->hasAttr("attention.force_reassociation");
    bool keyBenefit=!func->hasAttr("attention.disable_key_reassociation") &&
      (force || (mac(h,m,l,d)+mac(h,m,n,l))*1.05L < mac(h,n,d,l)+mac(h,m,n,d));
    bool valueBenefit=!func->hasAttr("attention.disable_value_reassociation") &&
      (force || (mac(h,m,l,n)+mac(h,m,dv,l))*1.05L < mac(h,n,dv,l)+mac(h,m,dv,n));
    if(!keyBenefit && !valueBenefit) return failure();
    auto loc=op->getLoc();Value newQ=q->value,newK=op->getOperand(1),newV=op->getOperand(2);
    if(keyBenefit) {
      Value contentQuery=newQ;
      Value rotaryQuery;
      if(rotaryKey) {
        SmallVector<OpFoldResult> offsets={rw.getIndexAttr(0),rw.getIndexAttr(0),rw.getIndexAttr(0)};
        SmallVector<OpFoldResult> sizes={rw.getIndexAttr(h),rw.getIndexAttr(m),rw.getIndexAttr(d)};
        SmallVector<OpFoldResult> strides(3,rw.getIndexAttr(1));
        contentQuery=rw.create<tensor::ExtractSliceOp>(loc,newQ,offsets,sizes,strides);
        offsets[2]=rw.getIndexAttr(d);sizes[2]=rw.getIndexAttr(rotaryWidth);
        rotaryQuery=rw.create<tensor::ExtractSliceOp>(loc,newQ,offsets,sizes,strides);
      }
      newQ=multiply(rw,loc,contentQuery,transpose(rw,loc,wk->value));newK=c->value;
      if(rotaryKey) {
        newQ=rw.create<tensor::ConcatOp>(loc,2,ValueRange{newQ,rotaryQuery});
        newK=rw.create<tensor::ConcatOp>(loc,2,ValueRange{newK,rotaryKey});
      }
      // tm_tensor.attention scales by inverse sqrt(query width). Preserve the
      // original D-based scale after changing that width to L.
      auto t=cast<RankedTensorType>(newQ.getType());
      Value empty=rw.create<tensor::EmptyOp>(loc,t.getShape(),t.getElementType());
      auto identity=AffineMap::getMultiDimIdentityMap(3,rw.getContext());
      newQ=rw.create<linalg::GenericOp>(loc,TypeRange{t},ValueRange{newQ},ValueRange{empty},
        ArrayRef<AffineMap>{identity,identity},
        ArrayRef<utils::IteratorType>{utils::IteratorType::parallel,utils::IteratorType::parallel,utils::IteratorType::parallel},
        [&](OpBuilder &b,Location here,ValueRange args){
          Value factor=b.create<arith::ConstantOp>(here,b.getF32FloatAttr(explicitScale?1.0:std::sqrt(static_cast<double>(l+rotaryWidth)/originalWidth)));
          Value result=b.create<arith::MulFOp>(here,args[0],factor);
          b.create<linalg::YieldOp>(here,result);
        }).getResult(0);
    }
    if(valueBenefit) newV=c->value;
    auto resultType=RankedTensorType::get({h,m,valueBenefit?l:dv},q->type.getElementType());
    Value empty=rw.create<tensor::EmptyOp>(loc,resultType.getShape(),resultType.getElementType());
    Value zero=rw.create<arith::ConstantOp>(loc,rw.getF32FloatAttr(0.0));
    Value newInit=rw.create<linalg::FillOp>(loc,ValueRange{zero},ValueRange{empty}).getResult(0);
    Operation *copy=rw.clone(*op);
    rw.modifyOpInPlace(copy,[&]{
      copy->setOperands(ValueRange{newQ,newK,newV,newInit});
      copy->getResult(0).setType(resultType);
      (void)copy->setPropertiesFromAttribute(originalProperties,
        [&] { return copy->emitError("invalid attention operand segments"); });
      copy->setAttr("attention.projection_placement",rw.getStringAttr(
        keyBenefit?(valueBenefit?"native-both":"native-key"):"native-value"));
    });
    Value result=copy->getResult(0);
    if(valueBenefit) result=multiply(rw,loc,result,wv->value);
    rw.replaceOp(op,result);return success();
  }
};
// CPU workaround for the independently reproduced wide-head online-attention
// mismatch in the pinned IREE build. This materializes scores and is explicitly
// selected by the caller; it is not a claim that GPU attention is defective.
struct CPUSafeAttentionPattern : RewritePattern {
  CPUSafeAttentionPattern(MLIRContext *ctx) : RewritePattern("tm_tensor.attention",1,ctx) {}
  LogicalResult matchAndRewrite(Operation *op,PatternRewriter &rw) const override {
    if(op->getNumOperands()!=4 || op->getNumResults()!=1 || op->getNumRegions()!=0 ||
       !attention_bounds::isZeroFill(op->getOperand(3)))return failure();
    auto q=matrix(op->getOperand(0)),k=matrix(op->getOperand(1)),v=matrix(op->getOperand(2));
    auto out=matrix(op->getResult(0));auto func=op->getParentOfType<func::FuncOp>();
    if(!q || !k || !v || !out)return failure();
    int64_t h=q->type.getDimSize(0),m=q->type.getDimSize(1),d=q->type.getDimSize(2),n=k->type.getDimSize(1),dv=v->type.getDimSize(2);
    if(d<=128 && !(func && func->hasAttr("attention.force_safe_attention_lowering")))return failure();
    if(k->type.getDimSize(0)!=h || v->type.getDimSize(0)!=h || v->type.getDimSize(1)!=n ||
       k->type.getDimSize(2)!=d || out->type.getShape()!=ArrayRef<int64_t>({h,m,dv}))return failure();
    auto properties=dyn_cast_or_null<DictionaryAttr>(op->getPropertiesAsAttribute());
    if(!properties)return failure();auto explicitScale=properties.getAs<FloatAttr>("scale");
    double scale=explicitScale?explicitScale.getValueAsDouble():1.0/std::sqrt(static_cast<double>(d));
    auto loc=op->getLoc();auto ctx=rw.getContext();
    auto d0=getAffineDimExpr(0,ctx),d1=getAffineDimExpr(1,ctx);
    auto identity=AffineMap::getMultiDimIdentityMap(3,ctx);
    auto row=AffineMap::get(3,0,{d0,d1},ctx);
    auto rowType=RankedTensorType::get({h,m},rw.getF32Type());
    auto scoreType=RankedTensorType::get({h,m,n},rw.getF32Type());
    SmallVector<utils::IteratorType> par(3,utils::IteratorType::parallel);
    SmallVector<utils::IteratorType> red={utils::IteratorType::parallel,utils::IteratorType::parallel,utils::IteratorType::reduction};
    Value scores=scoreContract(rw,loc,q->value,k->value);
    Value empty=rw.create<tensor::EmptyOp>(loc,scoreType.getShape(),rw.getF32Type());
    Value scaled=rw.create<linalg::GenericOp>(loc,TypeRange{scoreType},ValueRange{scores},ValueRange{empty},ArrayRef<AffineMap>{identity,identity},par,
      [&](OpBuilder &b,Location l,ValueRange args){
        Value factor=b.create<arith::ConstantOp>(l,b.getF32FloatAttr(scale));
        Value x=b.create<arith::MulFOp>(l,args[0],factor);b.create<linalg::YieldOp>(l,x);
      }).getResult(0);
    Value rowEmpty=rw.create<tensor::EmptyOp>(loc,rowType.getShape(),rw.getF32Type());
    Value negativeInfinity=rw.create<arith::ConstantOp>(loc,rw.getF32FloatAttr(-std::numeric_limits<float>::infinity()));
    Value maxInit=rw.create<linalg::FillOp>(loc,ValueRange{negativeInfinity},ValueRange{rowEmpty}).getResult(0);
    Value maxima=rw.create<linalg::GenericOp>(loc,TypeRange{rowType},ValueRange{scaled},ValueRange{maxInit},ArrayRef<AffineMap>{identity,row},red,
      [&](OpBuilder &b,Location l,ValueRange args){Value x=b.create<arith::MaximumFOp>(l,args[0],args[1]);b.create<linalg::YieldOp>(l,x);}).getResult(0);
    Value exponentials=rw.create<linalg::GenericOp>(loc,TypeRange{scoreType},ValueRange{scaled,maxima},ValueRange{empty},ArrayRef<AffineMap>{identity,row,identity},par,
      [&](OpBuilder &b,Location l,ValueRange args){Value diff=b.create<arith::SubFOp>(l,args[0],args[1]);Value x=b.create<math::ExpOp>(l,diff);b.create<linalg::YieldOp>(l,x);}).getResult(0);
    Value zero=rw.create<arith::ConstantOp>(loc,rw.getF32FloatAttr(0.0));
    Value sumInit=rw.create<linalg::FillOp>(loc,ValueRange{zero},ValueRange{rowEmpty}).getResult(0);
    Value sums=rw.create<linalg::GenericOp>(loc,TypeRange{rowType},ValueRange{exponentials},ValueRange{sumInit},ArrayRef<AffineMap>{identity,row},red,
      [&](OpBuilder &b,Location l,ValueRange args){Value x=b.create<arith::AddFOp>(l,args[0],args[1]);b.create<linalg::YieldOp>(l,x);}).getResult(0);
    Value probabilities=rw.create<linalg::GenericOp>(loc,TypeRange{scoreType},ValueRange{exponentials,sums},ValueRange{empty},ArrayRef<AffineMap>{identity,row,identity},par,
      [&](OpBuilder &b,Location l,ValueRange args){Value x=b.create<arith::DivFOp>(l,args[0],args[1]);b.create<linalg::YieldOp>(l,x);}).getResult(0);
    Value result=multiply(rw,loc,probabilities,v->value);
    result.getDefiningOp()->setAttr("attention.cpu_safe_lowering",rw.getUnitAttr());
    rw.replaceOp(op,result);return success();
  }
};
struct CPUSafeAttentionPass : PassWrapper<CPUSafeAttentionPass,OperationPass<func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(CPUSafeAttentionPass)
  StringRef getArgument() const final {return "attention-cpu-safe-lowering";}
  StringRef getDescription() const final {return "Lower wide-head unmasked f32 attention to verified standard contractions on CPU";}
  void getDependentDialects(DialectRegistry &r) const override {r.insert<linalg::LinalgDialect,tensor::TensorDialect,arith::ArithDialect,math::MathDialect>();}
  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());patterns.add<CPUSafeAttentionPattern>(&getContext());
    if(failed(applyPatternsAndFoldGreedily(getOperation(),std::move(patterns))))signalPassFailure();
  }
};
struct AttentionProjectionPass : PassWrapper<AttentionProjectionPass,OperationPass<func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(AttentionProjectionPass)
  StringRef getArgument() const final {return "attention-project-reassociate";}
  StringRef getDescription() const final {return "Place batched projections using contraction cost under explicit reassociation contract";}
  void getDependentDialects(DialectRegistry &r) const override {
    r.insert<linalg::LinalgDialect,tensor::TensorDialect,arith::ArithDialect>();
  }
  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());patterns.add<ProjectionPattern,NativeProjectionPattern>(&getContext());
    if(failed(applyPatternsAndFoldGreedily(getOperation(),std::move(patterns)))) signalPassFailure();
  }
};
}
void registerAttentionProjectionPass() {PassRegistration<AttentionProjectionPass>();PassRegistration<CPUSafeAttentionPass>();}
