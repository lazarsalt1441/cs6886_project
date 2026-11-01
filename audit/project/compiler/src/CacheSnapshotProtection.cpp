// SPDX-License-Identifier: MIT
// Preserve observed pre-update snapshots across IREE ABI import/export folding.
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/DenseMap.h"
using namespace mlir;
namespace {
struct CacheSnapshotProtection final
    : PassWrapper<CacheSnapshotProtection, OperationPass<func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(CacheSnapshotProtection)
  StringRef getArgument() const final { return "attention-protect-cache-snapshots"; }
  StringRef getDescription() const final {
    return "Preserve returned cache inputs also used by functional updates";
  }
  void runOnOperation() override {
    func::FuncOp fn = getOperation();
    if (fn.isDeclaration() || !llvm::hasSingleElement(fn.getBody())) return;
    llvm::DenseMap<Value, Value> snapshots;
    fn.walk([&](func::ReturnOp ret) {
      for (OpOperand &operand : ret->getOpOperands()) {
        auto arg = dyn_cast<BlockArgument>(operand.get());
        if (!arg || arg.getOwner() != &fn.front() ||
            !isa<RankedTensorType>(arg.getType())) continue;
        bool updated = llvm::any_of(arg.getUsers(), [&](Operation *user) {
          auto write = dyn_cast<tensor::InsertSliceOp>(user);
          return write && write.getDest() == arg;
        });
        if (!updated) continue;
        auto found = snapshots.find(arg);
        if (found == snapshots.end()) {
          OpBuilder builder(&fn.front(), fn.front().begin());
          OperationState state(fn.getLoc(), "util.optimization_barrier");
          state.addOperands(arg);
          state.addTypes(arg.getType());
          state.addAttribute("attention.cache_snapshot_guard", builder.getUnitAttr());
          Value value = builder.create(state)->getResult(0);
          found = snapshots.insert({arg, value}).first;
        }
        operand.set(found->second);
      }
    });
  }
};
// Phase-local repair for the demonstrated cast-hidden borrowed-storage case.
// This standalone form operates on registered IREE IR serialized generically;
// its output is reparsed and verified by the pinned IREE compiler.
struct BorrowedCacheCOW final
    : PassWrapper<BorrowedCacheCOW, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(BorrowedCacheCOW)
  StringRef getArgument() const final {
    return "attention-materialize-borrowed-cache-cow";
  }
  StringRef getDescription() const final {
    return "Materialize update COW across casts of non-consuming tensor imports";
  }
  void runOnOperation() override {
    getOperation().walk([&](Operation *op) {
      if (op->getName().getStringRef() != "stream.async.update" ||
          op->getNumOperands() != 6 || op->getNumResults() != 1 ||
          !op->getOperand(1).getType().isIndex() ||
          op->getOperand(0).getType() != op->getResult(0).getType()) return;
      Value target = op->getOperand(0), root = target;
      bool throughCast = false;
      while (Operation *producer = root.getDefiningOp()) {
        if (producer->getName().getStringRef() != "stream.async.cast" ||
            producer->getNumOperands() != 3 || producer->getNumResults() != 1)
          break;
        throughCast = true;
        root = producer->getOperand(0);
      }
      Operation *importOp = root.getDefiningOp();
      if (!throughCast || !importOp ||
          importOp->getName().getStringRef() != "stream.tensor.import") return;
      auto properties = dyn_cast_or_null<DictionaryAttr>(importOp->getPropertiesAsAttribute());
      if (importOp->hasAttr("consume") || (properties && properties.get("consume"))) return;
      OpBuilder builder(op);
      OperationState state(op->getLoc(), "stream.async.clone");
      state.addOperands({target, op->getOperand(1), op->getOperand(1)});
      state.addTypes(target.getType());
      state.addAttribute("attention.borrowed_cache_cow", builder.getUnitAttr());
      Operation *clone = builder.create(state);
      auto opProperties = dyn_cast_or_null<DictionaryAttr>(op->getPropertiesAsAttribute());
      Attribute affinity = op->getAttr("affinity");
      if (!affinity && opProperties) affinity = opProperties.get("affinity");
      if (affinity) {
        auto dict = builder.getDictionaryAttr({builder.getNamedAttr("affinity", affinity)});
        if (failed(clone->setPropertiesFromAttribute(dict, [&] {
              return clone->emitError("invalid clone affinity");
            }))) { signalPassFailure(); return; }
      }
      op->setOperand(0, clone->getResult(0));
    });
  }
};
} // namespace
void registerCacheSnapshotProtection() {
  PassRegistration<CacheSnapshotProtection>();
  PassRegistration<BorrowedCacheCOW>();
}
