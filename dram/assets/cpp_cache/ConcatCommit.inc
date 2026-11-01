// Factor only the committed selection; other attention users are unchanged.
struct FactorConcatCommit final:OpRewritePattern<linalg::GenericOp>{
 using OpRewritePattern::OpRewritePattern;
 LogicalResult matchAndRewrite(linalg::GenericOp op,PatternRewriter &rw)const override{
  if(op.getNumDpsInputs()!=3 || op.getNumDpsInits()!=1 || op.getNumResults()!=1 || op->hasAttr("lowering_config"))return failure();
  auto ty=dyn_cast<RankedTensorType>(op.getResult(0).getType());auto cond=dyn_cast<RankedTensorType>(op.getDpsInputs()[0].getType());
  if(!ty || ty.getEncoding() || ty.getRank()<1 || ty.getRank()>5 || !cond || cond.getRank()!=0 || !cond.getElementType().isInteger(1) || op.getNumLoops()!=ty.getRank() || !llvm::all_of(op.getIteratorTypesArray(),[](utils::IteratorType t){return t==utils::IteratorType::parallel;}))return failure();
  auto maps=op.getIndexingMapsArray();if(maps.size()!=4 || maps[0].getNumResults()!=0 || !maps[1].isIdentity() || !maps[2].isIdentity() || !maps[3].isIdentity())return failure();
  Block &body=op.getRegion().front();if(body.getOperations().size()!=2)return failure();
  auto select=dyn_cast<arith::SelectOp>(body.front());auto yield=dyn_cast<linalg::YieldOp>(body.back());
  if(!select || !yield || yield.getNumOperands()!=1 || yield.getOperand(0)!=select.getResult() || select.getCondition()!=body.getArgument(0) || select.getTrueValue()!=body.getArgument(1) || select.getFalseValue()!=body.getArgument(2))return failure();
  auto a=op.getDpsInputs()[1].getDefiningOp<tensor::ConcatOp>(),b=op.getDpsInputs()[2].getDefiningOp<tensor::ConcatOp>();
  if(!a || !b || a.getInputs().size()!=2 || b.getInputs().size()!=2 || a.getDim()!=b.getDim() || a.getInputs()[0]!=b.getInputs()[0] || a.getType()!=ty || b.getType()!=ty)return failure();
  Value lhs=a.getInputs()[1],rhs=b.getInputs()[1];auto update=dyn_cast<RankedTensorType>(lhs.getType());
  if(!update || !update.hasStaticShape() || update.getEncoding() || update.getRank()!=ty.getRank() || rhs.getType()!=update)return failure();
  for(int64_t d:update.getShape())if(d<=0)return failure();
  auto history=cast<RankedTensorType>(a.getInputs()[0].getType());
  if(history.hasStaticShape() && history.getDimSize(a.getDim())<update.getDimSize(a.getDim()))return failure();
  auto loc=op.getLoc();Value empty=rw.create<tensor::EmptyOp>(loc,update.getShape(),update.getElementType());
  auto small=rw.create<linalg::GenericOp>(loc,TypeRange{update},ValueRange{op.getDpsInputs()[0],lhs,rhs},ValueRange{empty},maps,op.getIteratorTypesArray(),
   [&](OpBuilder &builder,Location l,ValueRange args){Value v=builder.create<arith::SelectOp>(l,args[0],args[1],args[2]);builder.create<linalg::YieldOp>(l,v);});
  auto committed=rw.create<tensor::ConcatOp>(loc,a.getDim(),ValueRange{a.getInputs()[0],small.getResult(0)});
  committed->setAttr("attention.concat_commit_factored",rw.getUnitAttr());rw.replaceOp(op,committed.getResult());return success();
 }
};
