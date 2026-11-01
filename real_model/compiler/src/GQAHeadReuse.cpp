// SPDX-License-Identifier: MIT
// Eliminate the exact repeat_interleave lowering used by Torch-MLIR GQA.
#include <iterator>
#include <optional>
#include "llvm/Support/MathExtras.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/Passes.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"
using namespace mlir;
void registerAttentionProjectionPass();
void registerOnlineAttentionGuard();
void registerCacheReadForwarding();
void registerGatherIndexProjection();
void registerCacheGatherForwarding();
void registerCacheCommitFactoring();
void registerCacheUpdateCoalescing();
void registerCacheSnapshotProtection();
namespace {
bool groups(tensor::CollapseShapeOp op,
            std::initializer_list<std::initializer_list<int64_t>> wanted) {
  if (!op) return false;
  auto got = op.getReassociationIndices();
  if (got.size() != wanted.size()) return false;
  unsigned i=0;
  for (auto g : wanted) {
    if (ArrayRef<int64_t>(got[i++]) != ArrayRef<int64_t>(g.begin(),g.size())) return false;
  }
  return true;
}
struct Match { Value kv; int64_t b,h,r; bool transposed; };
std::optional<Match> matchRepeat(Value rhs) {
  auto outer=rhs.getDefiningOp<tensor::CollapseShapeOp>();
  if (!outer) return std::nullopt;
  Value expanded;
  bool transposed=false;
  if (groups(outer,{{0,1,2},{3},{4}})) {
    expanded=outer.getSrc();
  } else if (groups(outer,{{0,1},{2},{3}})) {
    auto t=outer.getSrc().getDefiningOp<linalg::TransposeOp>();
    if (!t || t.getPermutation()!=ArrayRef<int64_t>({0,1,3,2})) return std::nullopt;
    auto c=t.getInput().getDefiningOp<tensor::CollapseShapeOp>();
    if (!groups(c,{{0},{1,2},{3},{4}})) return std::nullopt;
    expanded=c.getSrc(); transposed=true;
  } else return std::nullopt;
  auto broadcast=expanded.getDefiningOp<linalg::GenericOp>();
  if (!broadcast || broadcast.getNumDpsInputs()!=1 || broadcast.getNumDpsInits()!=1 ||
      broadcast.getNumResults()!=1 || !broadcast.hasPureTensorSemantics()) return std::nullopt;
  auto out=dyn_cast<RankedTensorType>(expanded.getType());
  auto in=dyn_cast<RankedTensorType>(broadcast.getDpsInputs()[0].getType());
  if (!out || !in || out.getRank()!=5 || in.getRank()!=4 ||
      !out.hasStaticShape() || !in.hasStaticShape() || in.getElementType()!=out.getElementType()) return std::nullopt;
  auto maps=broadcast.getIndexingMapsArray();
  MLIRContext *ctx=broadcast.getContext();
  auto d=[&](unsigned i){return getAffineDimExpr(i,ctx);};
  if (maps[0]!=AffineMap::get(5,0,{d(0),d(1),d(3),d(4)},ctx) ||
      !maps[1].isIdentity()) return std::nullopt;
  for(auto it:broadcast.getIteratorTypesArray())
    if(it!=utils::IteratorType::parallel) return std::nullopt;
  Block &body=broadcast.getRegion().front();
  auto yield=dyn_cast<linalg::YieldOp>(body.getTerminator());
  if (!yield || body.getOperations().size()!=1 || yield.getNumOperands()!=1 ||
      yield.getOperand(0)!=body.getArgument(0)) return std::nullopt;
  auto s=out.getShape(); auto v=in.getShape();
  if(s[0]<=0 || s[1]<=0 || s[2]<=1 || s[3]<=0 || s[4]<=0 ||
     v[0]!=s[0] || v[1]!=s[1] || v[2]!=s[3] || v[3]!=s[4]) return std::nullopt;
  return Match{broadcast.getDpsInputs()[0],s[0],s[1],s[2],transposed};
}
struct Reuse : OpRewritePattern<linalg::BatchMatmulOp> {
  bool compact;
  Reuse(MLIRContext *ctx, bool compact=false) : OpRewritePattern(ctx), compact(compact) {}
  LogicalResult matchAndRewrite(linalg::BatchMatmulOp op, PatternRewriter &rw) const override {
    if(!op.hasPureTensorSemantics()) return failure();
    auto m=matchRepeat(op.getDpsInputs()[1]);
    if(!m) return failure();
    auto a=dyn_cast<RankedTensorType>(op.getDpsInputs()[0].getType());
    auto o=dyn_cast<RankedTensorType>(op.getDpsInits()[0].getType());
    auto kv=cast<RankedTensorType>(m->kv.getType());
    int64_t g, expandedBatch;
    if(llvm::MulOverflow(m->b,m->h,g) || llvm::MulOverflow(g,m->r,expandedBatch)) return failure();
    if(!a || !o || !a.hasStaticShape() || !o.hasStaticShape() ||
       a.getRank()!=3 || o.getRank()!=3 || !a.getElementType().isF32() ||
       a.getElementType()!=kv.getElementType() || o.getElementType()!=a.getElementType() ||
       a.getDimSize(0)!=expandedBatch || o.getDimSize(0)!=a.getDimSize(0) ||
       a.getDimSize(1)<=0 || a.getDimSize(2)<=0 || o.getDimSize(1)<=0 || o.getDimSize(2)<=0) return failure();
    // Do not clone arbitrary scalar regions/attributes (e.g. mixed precision casts).
    Block &old=op.getRegion().front();
    if(old.getOperations().size()!=3 || !isa<arith::MulFOp>(old.front()) ||
       !isa<arith::AddFOp>(*std::next(old.begin()))) return failure();
    auto mul=cast<arith::MulFOp>(old.front());
    auto add=cast<arith::AddFOp>(*std::next(old.begin()));
    auto yield=dyn_cast<linalg::YieldOp>(old.getTerminator());
    if(mul.getLhs()!=old.getArgument(0) || mul.getRhs()!=old.getArgument(1) ||
       add.getLhs()!=old.getArgument(2) || add.getRhs()!=mul.getResult() ||
       !yield || yield.getOperand(0)!=add.getResult() ||
       mul.getFastmath()!=arith::FastMathFlags::none ||
       add.getFastmath()!=arith::FastMathFlags::none) return failure();
    int64_t compactM;
    if(compact && llvm::MulOverflow(m->r,a.getDimSize(1),compactM)) return failure();
    auto loc=op.getLoc(); auto elem=a.getElementType();
    auto at=RankedTensorType::get({g,m->r,a.getDimSize(1),a.getDimSize(2)},elem);
    auto ot=RankedTensorType::get({g,m->r,o.getDimSize(1),o.getDimSize(2)},elem);
    SmallVector<ReassociationIndices> split={{0,1},{2},{3}};
    Value av=rw.create<tensor::ExpandShapeOp>(loc,at,op.getDpsInputs()[0],split);
    Value ov=rw.create<tensor::ExpandShapeOp>(loc,ot,op.getDpsInits()[0],split);
    auto kt=RankedTensorType::get({g,kv.getDimSize(2),kv.getDimSize(3)},elem);
    Value k=rw.create<tensor::CollapseShapeOp>(loc,kt,m->kv,
                      ArrayRef<ReassociationIndices>{{0,1},{2},{3}});
    auto d=[&](unsigned i){return getAffineDimExpr(i,rw.getContext());};
    SmallVector<AffineMap> maps={
      AffineMap::get(5,0,{d(0),d(1),d(2),d(4)},rw.getContext()),
      m->transposed ? AffineMap::get(5,0,{d(0),d(3),d(4)},rw.getContext())
                    : AffineMap::get(5,0,{d(0),d(4),d(3)},rw.getContext()),
      AffineMap::get(5,0,{d(0),d(1),d(2),d(3)},rw.getContext())};
    SmallVector<utils::IteratorType> it(4,utils::IteratorType::parallel);
    it.push_back(utils::IteratorType::reduction);
    RankedTensorType resultType=ot;
    if(compact) {
      // Merge adjacent (within-group, query-position) dimensions. The shared
      // KV operand remains compact, and all indexing maps stay projected.
      auto cat=RankedTensorType::get({g,compactM,a.getDimSize(2)},elem);
      resultType=RankedTensorType::get({g,compactM,o.getDimSize(2)},elem);
      SmallVector<ReassociationIndices> merge={{0},{1,2},{3}};
      av=rw.create<tensor::CollapseShapeOp>(loc,cat,av,merge);
      ov=rw.create<tensor::CollapseShapeOp>(loc,resultType,ov,merge);
      maps={AffineMap::get(4,0,{d(0),d(1),d(3)},rw.getContext()),
            m->transposed ? AffineMap::get(4,0,{d(0),d(2),d(3)},rw.getContext())
                          : AffineMap::get(4,0,{d(0),d(3),d(2)},rw.getContext()),
            AffineMap::get(4,0,{d(0),d(1),d(2)},rw.getContext())};
      it=SmallVector<utils::IteratorType>(3,utils::IteratorType::parallel);
      it.push_back(utils::IteratorType::reduction);
    }
    auto result=rw.create<linalg::GenericOp>(loc,TypeRange{resultType},ValueRange{av,k},ValueRange{ov},maps,it,
      [&](OpBuilder &b,Location l,ValueRange args){
        Value product=b.create<arith::MulFOp>(l,args[0],args[1]);
        Value sum=b.create<arith::AddFOp>(l,args[2],product);
        b.create<linalg::YieldOp>(l,sum);
      });
    Value resultValue=result.getResult(0);
    if(compact)
      resultValue=rw.create<tensor::ExpandShapeOp>(loc,ot,resultValue,
                       ArrayRef<ReassociationIndices>{{0},{1,2},{3}});
    Value collapsed=rw.create<tensor::CollapseShapeOp>(loc,o,resultValue,split);
    rw.replaceOp(op,collapsed);
    return success();
  }
};
struct GQAHeadReusePass : PassWrapper<GQAHeadReusePass,OperationPass<func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(GQAHeadReusePass)
  StringRef getArgument() const final {return "gqa-head-reuse";}
  StringRef getDescription() const final {return "Replace repeated KV heads with grouped contractions";}
  void getDependentDialects(DialectRegistry &r) const override {
    r.insert<linalg::LinalgDialect,tensor::TensorDialect,arith::ArithDialect>();
  }
  void runOnOperation() override {
    RewritePatternSet patterns(&getContext()); patterns.add<Reuse>(&getContext());
    if(failed(applyPatternsAndFoldGreedily(getOperation(),std::move(patterns)))) signalPassFailure();
  }
};
struct GQACompactHeadReusePass : PassWrapper<GQACompactHeadReusePass,OperationPass<func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(GQACompactHeadReusePass)
  StringRef getArgument() const final {return "gqa-compact-head-reuse";}
  StringRef getDescription() const final {return "Group KV heads and merge query loops";}
  void getDependentDialects(DialectRegistry &r) const override {
    r.insert<linalg::LinalgDialect,tensor::TensorDialect,arith::ArithDialect>();
  }
  void runOnOperation() override {
    RewritePatternSet patterns(&getContext()); patterns.add<Reuse>(&getContext(),true);
    if(failed(applyPatternsAndFoldGreedily(getOperation(),std::move(patterns)))) signalPassFailure();
  }
};

}
void registerStableTrig();
int main(int argc,char **argv) {
  DialectRegistry registry;
  registry.insert<func::FuncDialect,linalg::LinalgDialect,tensor::TensorDialect,
                  arith::ArithDialect,math::MathDialect,scf::SCFDialect>();
  registerStableTrig(); registerCacheSnapshotProtection(); registerCacheUpdateCoalescing(); registerCacheCommitFactoring(); registerCacheGatherForwarding(); registerGatherIndexProjection(); registerCacheReadForwarding(); registerAttentionProjectionPass(); registerOnlineAttentionGuard(); registerTransformsPasses(); PassRegistration<GQAHeadReusePass>(); PassRegistration<GQACompactHeadReusePass>();
  return asMainReturnCode(MlirOptMain(argc,argv,"GQA KV head reuse optimizer\n",registry));
}
