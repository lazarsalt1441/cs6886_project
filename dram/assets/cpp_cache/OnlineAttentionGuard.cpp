// SPDX-License-Identifier: MIT
// Online softmax may partition KV tokens (K2), but cannot treat partial QK
// feature reductions (K1) as independent softmax updates. This pass guards a
// pinned CPU lowering configuration before attention decomposition.
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/Support/Regex.h"
using namespace mlir;
namespace {
struct OnlineAttentionGuard : PassWrapper<OnlineAttentionGuard,OperationPass<ModuleOp>> {
 MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(OnlineAttentionGuard)
 StringRef getArgument() const final {return "attention-guard-online-reductions";}
 StringRef getDescription() const final {return "Forbid CPU feature-axis tiling across an online-softmax update";}
 void runOnOperation() override {
  getOperation().walk([&](Operation *op){
   if(op->getName().getStringRef()!="iree_linalg_ext.online_attention")return;
   auto properties=dyn_cast_or_null<DictionaryAttr>(op->getPropertiesAsAttribute());
   ArrayAttr maps=op->getAttrOfType<ArrayAttr>("indexing_maps");
   if(!maps && properties)maps=properties.getAs<ArrayAttr>("indexing_maps");
   auto config=dyn_cast_or_null<OpaqueAttr>(op->getAttr("lowering_config"));
   if(!maps || maps.size()!=7 || op->getNumOperands()!=7 ||
      op->getNumResults()!=3 || !config ||
      config.getDialectNamespace().getValue()!="iree_cpu")return;
   SmallVector<AffineMap> indexing;
   for(Attribute attr:maps){auto m=dyn_cast<AffineMapAttr>(attr);if(!m)return;indexing.push_back(m.getValue());}
   unsigned rank=indexing[0].getNumDims();
   // This bridge accepts generic operations, so do not rely on an IREE dialect
   // verifier to enforce its operand/map/region contracts for us.
   SmallVector<int64_t> extents(rank,-1);
   for(unsigned i=0;i<indexing.size();++i) {
    AffineMap map=indexing[i];
    if(map.getNumDims()!=rank || map.getNumSymbols()!=0 ||
       !map.isProjectedPermutation())return;
    auto tensor=dyn_cast<RankedTensorType>(op->getOperand(i).getType());
    if(!tensor) {
     if(!op->getOperand(i).getType().isF32() || map.getNumResults()!=0)return;
     continue;
    }
    if(!tensor.hasStaticShape() || !tensor.getElementType().isF32() ||
       map.getNumResults()!=static_cast<unsigned>(tensor.getRank()))return;
    for(auto [axis,expr]:llvm::enumerate(map.getResults())) {
     unsigned dim=cast<AffineDimExpr>(expr).getPosition();
     int64_t extent=tensor.getDimSize(axis);
     if(extent<=0 || (extents[dim]!=-1 && extents[dim]!=extent))return;
     extents[dim]=extent;
    }
   }
   for(unsigned i=0;i<3;++i)
    if(op->getResult(i).getType()!=op->getOperand(i+4).getType())return;
   if(op->getNumRegions()!=1 || !llvm::hasSingleElement(op->getRegion(0)))return;
   Block &scoreBody=op->getRegion(0).front();
   if(scoreBody.getNumArguments()!=1 || !scoreBody.getArgument(0).getType().isF32() ||
      !llvm::hasSingleElement(scoreBody))return;
   Operation &yield=scoreBody.front();
   if(yield.getName().getStringRef()!="iree_linalg_ext.yield" ||
      yield.getNumOperands()!=1 || yield.getOperand(0)!=scoreBody.getArgument(0))return;
   auto dims=[](AffineMap map){llvm::SmallSet<unsigned,8> found;for(AffineExpr e:map.getResults())if(auto d=dyn_cast<AffineDimExpr>(e))found.insert(d.getPosition());return found;};
   auto query=dims(indexing[0]),key=dims(indexing[1]),value=dims(indexing[2]),output=dims(indexing[4]);
   SmallVector<unsigned> feature;
   for(unsigned d:query)if(key.contains(d) && !value.contains(d) && !output.contains(d))feature.push_back(d);
   if(feature.empty())return;
   auto queryType=dyn_cast<RankedTensorType>(op->getOperand(0).getType());
   if(!queryType || !queryType.hasStaticShape() || !queryType.getElementType().isF32())return;
   // Avoid creating unbounded vectors: large/dynamic features need a different
   // staged decomposition, not silent suppression of the resource bound.
   int64_t featureExtent=1;
   for(unsigned f:feature){
    bool found=false;
    for(auto [i,e]:llvm::enumerate(indexing[0].getResults()))if(auto d=dyn_cast<AffineDimExpr>(e))
     if(d.getPosition()==f){int64_t extent=queryType.getDimSize(i);if(extent<=0 || extent>1024 || featureExtent>1024/extent)return;featureExtent*=extent;found=true;}
    if(!found)return;
   }
   StringRef data=config.getAttrData();SmallVector<StringRef> match;
   llvm::Regex pattern("vector_reduction[ ]*=[ ]*\\[([^]]*)\\]");
   if(!pattern.match(data,&match))return;
   SmallVector<StringRef> fields;match[1].split(fields,',');
   if(fields.size()!=rank)return;
   SmallVector<int64_t> tiles;
   for(StringRef field:fields){int64_t tile;if(field.trim().getAsInteger(10,tile) || tile<0)return;tiles.push_back(tile);}
   bool changed=false;for(unsigned f:feature)if(tiles[f]){tiles[f]=0;changed=true;}
   if(!changed)return;
   // Bound the complete vectorized QK contraction volume, not each operand
   // independently: later contraction lowering may form an outer-product view.
   SmallVector<StringRef> commonMatch;
   llvm::Regex commonPattern("vector_common_parallel[ ]*=[ ]*\\[([^]]*)\\]");
   SmallVector<int64_t> common;
   if(!commonPattern.match(data,&commonMatch))return;
   SmallVector<StringRef> pieces;commonMatch[1].split(pieces,',');
   if(pieces.size()!=rank)return;
   for(StringRef piece:pieces){int64_t tile;if(piece.trim().getAsInteger(10,tile) || tile<0)return;common.push_back(tile);}
   SmallVector<unsigned> rows,sequence;
   for(unsigned d:query)if(output.contains(d)){rows.push_back(d);if(common[d]==0)common[d]=1;}
   for(unsigned d:key)if(value.contains(d) && !query.contains(d) && !output.contains(d)){
    sequence.push_back(d);if(tiles[d]==0)tiles[d]=1;
   }
   auto volume=[&]{long double v=featureExtent;for(unsigned d:rows)v*=common[d];for(unsigned d:sequence)v*=tiles[d];return v;};
   unsigned attempts=0;
   while(volume()>2048.0L && attempts++<4096) {
    int64_t largest=1;int64_t *selected=nullptr;
    for(unsigned d:rows)if(common[d]>largest){largest=common[d];selected=&common[d];}
    for(unsigned d:sequence)if(tiles[d]>largest){largest=tiles[d];selected=&tiles[d];}
    if(!selected)return;
    *selected=std::max<int64_t>(1,*selected/2);
   }
   if(volume()>2048.0L)return;
   std::string replacement="vector_reduction = [";
   for(unsigned i=0;i<tiles.size();++i){if(i)replacement+=", ";replacement+=std::to_string(tiles[i]);}
   replacement+="]";
   std::string body=data.str();size_t offset=match[0].data()-data.data();body.replace(offset,match[0].size(),replacement);
   if(!common.empty()) {
    std::string commonReplacement="vector_common_parallel = [";
    for(unsigned i=0;i<common.size();++i){if(i)commonReplacement+=", ";commonReplacement+=std::to_string(common[i]);}
    commonReplacement+="]";SmallVector<StringRef> current;
    if(commonPattern.match(body,&current)) {
      size_t at=current[0].data()-body.data();body.replace(at,current[0].size(),commonReplacement);
    }
   }
   Attribute updated=OpaqueAttr::get(config.getDialectNamespace(),body,config.getType());
   op->setAttr("lowering_config",updated);
   op->setAttr("attention.feature_reduction_guard",UnitAttr::get(&getContext()));
  });
 }
};
}
void registerOnlineAttentionGuard() {PassRegistration<OnlineAttentionGuard>();}
