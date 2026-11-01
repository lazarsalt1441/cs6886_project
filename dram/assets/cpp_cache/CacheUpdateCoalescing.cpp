// SPDX-License-Identifier: MIT
// Coalesce adjacent unobserved update chains before cache-read forwarding.
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
using namespace mlir;
namespace {
struct CoalesceUpdates final : OpRewritePattern<tensor::InsertSliceOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(tensor::InsertSliceOp outer,
                                PatternRewriter &rw) const override {
    auto inner = outer.getDest().getDefiningOp<tensor::InsertSliceOp>();
    if (!inner || !inner.getResult().hasOneUse()) return failure();
    auto dst = cast<RankedTensorType>(outer.getType());
    auto a = cast<RankedTensorType>(inner.getSource().getType());
    auto b = cast<RankedTensorType>(outer.getSource().getType());
    if (!dst.hasStaticShape() || !a.hasStaticShape() || !b.hasStaticShape() ||
        dst.getEncoding() || a.getEncoding() || b.getEncoding() ||
        dst.getRank() < 1 || dst.getRank() > 5 ||
        a.getRank() != dst.getRank() || b.getRank() != dst.getRank())
      return failure();
    for (int64_t size : a.getShape()) if (size <= 0) return failure();
    for (int64_t size : b.getShape()) if (size <= 0) return failure();
    auto ao = inner.getMixedOffsets(), bo = outer.getMixedOffsets();
    auto as = inner.getMixedStrides(), bs = outer.getMixedStrides();
    for (int64_t i = 0; i < dst.getRank(); ++i) {
      auto sa = getConstantIntValue(as[i]), sb = getConstantIntValue(bs[i]);
      if (!sa || !sb || *sa != 1 || *sb != 1) return failure();
      if (i == 0) continue;
      if (a.getDimSize(i) != b.getDimSize(i)) return failure();
      auto oa = getConstantIntValue(ao[i]), ob = getConstantIntValue(bo[i]);
      if (oa && ob) { if (*oa != *ob) return failure(); }
      else if (ao[i] != bo[i]) return failure();
    }
    auto first = getConstantIntValue(ao[0]);
    auto second = getConstantIntValue(bo[0]);
    int64_t na = a.getDimSize(0), nb = b.getDimSize(0), pool = dst.getDimSize(0);
    if (!first || !second || *first < 0 || *second < 0 ||
        na > pool || nb > pool || *first > pool - na || *second > pool - nb)
      return failure();
    bool append = *second == *first + na;
    bool prepend = *first == *second + nb;
    if (!append && !prepend) return failure();
    // Target copy-prone forks/observations, not already-in-place single-use
    // cache updates. Keep the temporary joined update small relative to pool.
    if (inner.getDest().hasOneUse() || na + nb > pool / 4) return failure();
    // The disjoint adjacent regions fit the same destination, so their sum
    // cannot overflow and preserves original ordering even when prepending.
    SmallVector<Value> sources;
    auto collect = [&](auto &&self, Value value) -> void {
      auto concat = value.getDefiningOp<tensor::ConcatOp>();
      if (concat && concat.getDim() == 0) {
        for (Value input : concat.getInputs()) self(self, input);
      } else {
        sources.push_back(value);
      }
    };
    collect(collect, append ? inner.getSource() : outer.getSource());
    collect(collect, append ? outer.getSource() : inner.getSource());
    Value joined = rw.create<tensor::ConcatOp>(outer.getLoc(), 0, sources);
    auto sizes = inner.getMixedSizes();
    ao[0] = rw.getIndexAttr(append ? *first : *second);
    sizes[0] = rw.getIndexAttr(na + nb);
    auto result = rw.create<tensor::InsertSliceOp>(outer.getLoc(), joined,
        inner.getDest(), ao, sizes, as);
    result->setAttr("attention.cache_updates_coalesced", rw.getUnitAttr());
    rw.replaceOp(outer, result.getResult());
    return success();
  }
};
struct CacheUpdateCoalescing final : PassWrapper<CacheUpdateCoalescing,
                                               OperationPass<>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(CacheUpdateCoalescing)
  StringRef getArgument() const final { return "attention-coalesce-cache-updates"; }
  StringRef getDescription() const final {
    return "Coalesce adjacent single-use KV update chains into one small update";
  }
  void getDependentDialects(DialectRegistry &r) const override {
    r.insert<tensor::TensorDialect>();
  }
  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    patterns.add<CoalesceUpdates>(&getContext());
    if (failed(applyPatternsAndFoldGreedily(getOperation(), std::move(patterns))))
      signalPassFailure();
  }
};
} // namespace
void registerCacheUpdateCoalescing() { PassRegistration<CacheUpdateCoalescing>(); }
