"""Explicit upstream gather-form baseline (known IREE #24374 opportunity).
This helper is a comparator, not the new C++ cache forwarding transformation.
"""
from iree.compiler import ir
from pathlib import Path

def walk(op):
 yield op
 for r in op.regions:
  for b in r.blocks:
   for c in b.operations:yield from walk(c.operation)
def lift(source,destination):
 count=0
 with ir.Context(),ir.Location.unknown():
  module=ir.Module.parse(Path(source).read_text())
  for op in list(walk(module.operation)):
   if op.name!='linalg.generic' or len(op.operands)!=2 or len(op.results)!=1:continue
   region=op.regions[0];block=region.blocks[0];ops=list(block.operations);extracts=[x.operation for x in ops if x.operation.name=='tensor.extract']
   if len(extracts)!=1 or len(ops) not in [4,5]:continue
   extract=extracts[0];idx=op.operands[0];out=op.operands[1];pool=extract.operands[0]
   it=ir.RankedTensorType(idx.type);ot=ir.RankedTensorType(out.type);pt=ir.RankedTensorType(pool.type)
   if it.rank!=1 or ot.rank!=3 or pt.rank!=3 or any(d<1 for d in [*it.shape,*ot.shape,*pt.shape]):continue
   if list(ot.shape[1:])!=list(pt.shape[1:]):continue
   # Restrict this comparator to the exact frontend index_select fixture.
   text=op.get_asm();assert 'index_cast' in text and 'linalg.index 2' in text
   assert pt.shape[1]==1 or 'linalg.index 1' in text
   n=it.shape[0];expanded=f'tensor<{n}x1x{it.element_type}>'
   temp=ir.Module.parse(f'''module {{ func.func @temporary(%pool: {pt}, %idx: {it}, %out: {ot}) -> {ot} {{
    %ii = tensor.expand_shape %idx [[0, 1]] output_shape [{n}, 1] : {it} into {expanded}
    %r = iree_linalg_ext.gather dimension_map = [0] ins(%pool, %ii : {pt}, {expanded}) outs(%out : {ot}) -> {ot}
    return %r : {ot}
   }} }}''')
   tb=temp.body.operations[0].regions[0].blocks[0];t_ops=list(tb.operations)
   with ir.InsertionPoint(op):
    expand=t_ops[0].operation.clone();expand.operands[0]=idx
    gather=t_ops[1].operation.clone();gather.operands[0]=pool;gather.operands[1]=expand.results[0];gather.operands[2]=out
   op.results[0].replace_all_uses_with(gather.results[0]);op.erase();count+=1
  module.operation.verify();Path(destination).write_text(str(module))
 return count
if __name__=='__main__':
 import sys
 print('lifted',lift(sys.argv[1],sys.argv[2]))
