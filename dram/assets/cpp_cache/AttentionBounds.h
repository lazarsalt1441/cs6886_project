// SPDX-License-Identifier: MIT
// Conditional absolute-value analysis used only in opt-in range-safe mode.
// Argument bounds are caller contracts, not observations or runtime checks.
#pragma once
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Matchers.h"
#include <optional>
#include <cmath>
#include <limits>
namespace attention_bounds {
using namespace mlir;
using Bound=std::optional<long double>;
inline Bound constant(Value v) {
 Attribute a;if(!matchPattern(v,m_Constant(&a)))return std::nullopt;
 auto f=dyn_cast<FloatAttr>(a);if(!f)return std::nullopt;
 double x=f.getValueAsDouble();if(!std::isfinite(x))return std::nullopt;
 return std::abs(static_cast<long double>(x));
}
inline bool isZeroFill(Value v) {
 auto f=v.getDefiningOp<linalg::FillOp>();return f && constant(f.getInputs()[0])==Bound(0.0L);
}
inline Value stripShape(Value v) {
 for(unsigned i=0;i<8;++i) {
  if(auto e=v.getDefiningOp<tensor::ExpandShapeOp>()){v=e.getSrc();continue;}
  if(auto c=v.getDefiningOp<tensor::CollapseShapeOp>()){v=c.getSrc();continue;}
  break;
 }
 return v;
}
inline bool soleScalar(linalg::GenericOp g,StringRef name) {
 return g && g.getRegion().front().getOperations().size()==2 &&
   g.getRegion().front().front().getName().getStringRef()==name;
}
inline bool yields(linalg::GenericOp g,Value value) {
 auto y=dyn_cast<linalg::YieldOp>(g.getRegion().front().getTerminator());
 return y && y.getNumOperands()==1 && y.getOperand(0)==value;
}
inline Bound absolute(Value v,func::FuncOp func,unsigned depth=0);
// A recognized stable-softmax row has a maximum element exp(0)=1, a sum at
// least one, and probabilities in [0,1]. Unknown/all-masked rows are not inferred.
inline Bound probability(linalg::GenericOp div,func::FuncOp func,unsigned depth) {
 if(!soleScalar(div,"arith.divf") || div.getNumDpsInputs()!=2 || div.getNumResults()!=1)
  return std::nullopt;
 Block &db=div.getRegion().front();auto d=cast<arith::DivFOp>(db.front());
 if(d.getLhs()!=db.getArgument(0) || d.getRhs()!=db.getArgument(1) || !yields(div,d.getResult()))return std::nullopt;
 Value numerator=div.getDpsInputs()[0];
 auto exp=numerator.getDefiningOp<linalg::GenericOp>();
 auto sum=stripShape(div.getDpsInputs()[1]).getDefiningOp<linalg::GenericOp>();
 if(!soleScalar(exp,"math.exp") || exp.getNumDpsInputs()!=1 ||
    !soleScalar(sum,"arith.addf") || sum.getNumDpsInputs()!=1 ||
    sum.getDpsInputs()[0]!=numerator || !isZeroFill(sum.getDpsInits()[0]))return std::nullopt;
 auto expOp=cast<math::ExpOp>(exp.getRegion().front().front());
 auto add=cast<arith::AddFOp>(sum.getRegion().front().front());
 if(expOp.getOperand()!=exp.getRegion().front().getArgument(0) || !yields(exp,expOp.getResult()) ||
    !yields(sum,add.getResult()) || add.getLhs()!=sum.getRegion().front().getArgument(0) ||
    add.getRhs()!=sum.getRegion().front().getArgument(1))return std::nullopt;
 auto sub=exp.getDpsInputs()[0].getDefiningOp<linalg::GenericOp>();
 if(!soleScalar(sub,"arith.subf") || sub.getNumDpsInputs()!=2)return std::nullopt;
 auto subtract=cast<arith::SubFOp>(sub.getRegion().front().front());
 if(subtract.getLhs()!=sub.getRegion().front().getArgument(0) ||
    subtract.getRhs()!=sub.getRegion().front().getArgument(1) || !yields(sub,subtract.getResult()))return std::nullopt;
 auto maxValue=dyn_cast<OpResult>(stripShape(sub.getDpsInputs()[1]));
 if(!maxValue || maxValue.getResultNumber()!=0)return std::nullopt;
 auto maximum=dyn_cast<linalg::GenericOp>(maxValue.getOwner());
 if(!maximum || maximum.getNumDpsInputs()!=1 || maximum.getDpsInputs()[0]!=sub.getDpsInputs()[0])return std::nullopt;
 auto init=maximum.getDpsInits()[0].getDefiningOp<linalg::FillOp>();Attribute initAttr;
 if(!init || !matchPattern(init.getInputs()[0],m_Constant(&initAttr)))return std::nullopt;
 auto inf=dyn_cast<FloatAttr>(initAttr);
 if(!inf || !inf.getValue().isInfinity() || !inf.getValue().isNegative())return std::nullopt;
 bool hasMax=false;Block &maxBody=maximum.getRegion().front();
 auto maxYield=dyn_cast<linalg::YieldOp>(maxBody.getTerminator());
 for(Operation &op:maxBody)if(auto maximumOp=dyn_cast<arith::MaximumFOp>(op))
  if(maximumOp.getLhs()==maxBody.getArgument(0) && maximumOp.getRhs()==maxBody.getArgument(1) &&
     maxYield && maxYield.getOperand(0)==maximumOp.getResult())hasMax=true;
 if(!hasMax)return std::nullopt;
 auto ctx=func.getContext();auto d0=getAffineDimExpr(0,ctx),d1=getAffineDimExpr(1,ctx);
 auto identity=AffineMap::getMultiDimIdentityMap(3,ctx);
 auto maxMaps=maximum.getIndexingMapsArray(),sumMaps=sum.getIndexingMapsArray();
 auto expMaps=exp.getIndexingMapsArray(),subMaps=sub.getIndexingMapsArray(),divMaps=div.getIndexingMapsArray();
 auto broadcast=AffineMap::get(3,0,{d0,d1,getAffineConstantExpr(0,ctx)},ctx);
 if(expMaps!=SmallVector<AffineMap>{identity,identity} ||
    subMaps!=SmallVector<AffineMap>{identity,broadcast,identity} ||
    divMaps!=SmallVector<AffineMap>{identity,broadcast,identity})return std::nullopt;
 auto maxIt=maximum.getIteratorTypesArray(),sumIt=sum.getIteratorTypesArray();
 if(maxIt!=SmallVector<utils::IteratorType>{utils::IteratorType::parallel,utils::IteratorType::parallel,utils::IteratorType::reduction} ||
    sumIt!=maxIt || maxMaps[0]!=identity || sumMaps[0]!=identity ||
    maxMaps[1]!=AffineMap::get(3,0,{d0,d1},ctx) ||
    sumMaps[1]!=AffineMap::get(3,0,{d0,d1,getAffineConstantExpr(0,ctx)},ctx))return std::nullopt;
 auto scores=absolute(sub.getDpsInputs()[0],func,depth+1);
 if(!scores || *scores>std::numeric_limits<float>::max()/16.0L)return std::nullopt;
 return 1.0L;
}
inline Bound absolute(Value v,func::FuncOp func,unsigned depth) {
 if(depth>24)return std::nullopt;
 if(auto arg=dyn_cast<BlockArgument>(v)) {
  if(arg.getOwner()!=&func.getBody().front())return std::nullopt;
  auto bounds=func->getAttrOfType<ArrayAttr>("attention.input_abs_bounds");
  if(!bounds || arg.getArgNumber()>=bounds.size())return std::nullopt;
  auto f=dyn_cast<FloatAttr>(bounds[arg.getArgNumber()]);
  if(!f || !std::isfinite(f.getValueAsDouble()) || f.getValueAsDouble()<0)return std::nullopt;
  return f.getValueAsDouble();
 }
 if(auto c=constant(v))return c;
 if(auto e=v.getDefiningOp<tensor::ExpandShapeOp>())return absolute(e.getSrc(),func,depth+1);
 if(auto c=v.getDefiningOp<tensor::CollapseShapeOp>())return absolute(c.getSrc(),func,depth+1);
 if(auto e=v.getDefiningOp<tensor::ExtractSliceOp>())return absolute(e.getSource(),func,depth+1);
 if(auto t=v.getDefiningOp<linalg::TransposeOp>())return absolute(t.getInput(),func,depth+1);
 if(auto f=v.getDefiningOp<linalg::FillOp>())return constant(f.getInputs()[0]);
 if(auto c=v.getDefiningOp<tensor::ConcatOp>()) {
  long double bound=0;for(Value x:c.getInputs()){auto b=absolute(x,func,depth+1);if(!b)return std::nullopt;bound=std::max(bound,*b);}return bound;
 }
 if(auto b=v.getDefiningOp<linalg::BatchMatmulOp>()) {
  auto a=absolute(b.getDpsInputs()[0],func,depth+1),c=absolute(b.getDpsInputs()[1],func,depth+1);
  if(!a || !c || !isZeroFill(b.getDpsInits()[0]))return std::nullopt;
  auto t=dyn_cast<RankedTensorType>(b.getDpsInputs()[0].getType());
  if(!t || !t.hasStaticShape() || t.getDimSize(2)<=0 || t.getDimSize(2)>1000000)return std::nullopt;
  return 2.0L*t.getDimSize(2)*(*a)*(*c); // Conservative rounding margin.
 }
 auto g=v.getDefiningOp<linalg::GenericOp>();
 if(!g || g.getNumResults()!=1)return std::nullopt;
 if(auto p=probability(g,func,depth))return p;
 for(auto it:g.getIteratorTypesArray())if(it!=utils::IteratorType::parallel)return std::nullopt;
 Block &body=g.getRegion().front();DenseMap<Value,long double> values;
 for(auto [i,input]:llvm::enumerate(g.getDpsInputs())) {
  auto b=absolute(input,func,depth+1);if(!b)return std::nullopt;values[body.getArgument(i)]=*b;
 }
 for(Operation &op:body) {
  if(auto y=dyn_cast<linalg::YieldOp>(op)) {
   auto found=values.find(y.getOperand(0));if(found==values.end())return std::nullopt;return found->second;
  }
  if(op.getNumResults()!=1)return std::nullopt;
  if(auto c=constant(op.getResult(0))){values[op.getResult(0)]=*c;continue;}
  if(isa<arith::TruncFOp,arith::ExtFOp>(op)) {
   auto found=values.find(op.getOperand(0));
   if(found==values.end()) {auto b=absolute(op.getOperand(0),func,depth+1);if(!b)return std::nullopt;values[op.getResult(0)]=2*(*b);}
   else values[op.getResult(0)]=2*found->second;
   continue;
  }
  if(op.getNumOperands()!=2)return std::nullopt;
  auto a=values.find(op.getOperand(0)),b=values.find(op.getOperand(1));
  if(a==values.end()) {auto v=constant(op.getOperand(0));if(!v)return std::nullopt;values[op.getOperand(0)]=*v;}
  if(b==values.end()) {auto v=constant(op.getOperand(1));if(!v)return std::nullopt;values[op.getOperand(1)]=*v;}
  a=values.find(op.getOperand(0));b=values.find(op.getOperand(1));
  if(isa<arith::MulFOp>(op))values[op.getResult(0)]=2*a->second*b->second;
  else if(isa<arith::AddFOp,arith::SubFOp>(op))values[op.getResult(0)]=2*(a->second+b->second);
  else return std::nullopt;
 }
 return std::nullopt;
}
inline bool safeTriple(func::FuncOp func,Value a,Value b,Value c,int64_t innerA,int64_t innerB) {
 if(!func->hasAttr("attention.require_range_safety"))return true;
 auto aa=absolute(a,func),bb=absolute(b,func),cc=absolute(c,func);
 if(!aa || !bb || !cc || innerA<=0 || innerB<=0 || innerA>1000000 || innerB>1000000)return false;
 long double limit=std::numeric_limits<float>::max()/16.0L;
 long double ab=(*aa)*(*bb)*innerA,bc=(*bb)*(*cc)*innerB;
 long double total=(*aa)*(*bb)*(*cc)*innerA*innerB;
 return std::isfinite(ab) && std::isfinite(bc) && std::isfinite(total) &&
   *aa<=limit && *bb<=limit && *cc<=limit && ab<=limit && bc<=limit && total<=limit;
}
}
