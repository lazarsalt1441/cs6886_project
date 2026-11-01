"""Structured bridge: IREE24 configured tensors -> MLIR19 C++ guard -> IREE24.
HAL memref memory-space interfaces cannot be parsed by a foreign SDK, so only
compatible tensor operations are passed across, and only their config is spliced.
"""
import sys,subprocess,os
from pathlib import Path
from iree.compiler import ir
from iree.compiler.dialects import func

def walk(op):
 yield op
 for region in op.regions:
  for block in region.blocks:
   for child in block.operations:yield from walk(child.operation)

def guard(source,destination,scratch):
 scratch=Path(scratch);scratch.mkdir(parents=True,exist_ok=True)
 with ir.Context(),ir.Location.unknown():
  module=ir.Module.parse(Path(source).read_text());count=0
  for op in list(walk(module.operation)):
   if op.name!='iree_linalg_ext.online_attention' or 'lowering_config' not in op.attributes:continue
   one=ir.Module.create()
   with ir.InsertionPoint(one.body):f=func.FuncOp('guard',([v.type for v in op.operands],[v.type for v in op.results]))
   block=f.add_entry_block()
   with ir.InsertionPoint(block):
    copy=op.clone()
    for i,a in enumerate(block.arguments):copy.operands[i]=a
    func.ReturnOp(list(copy.results))
   input_file=scratch/f'input-{count}.mlir';output_file=scratch/f'output-{count}.mlir'
   input_file.write_text(one.operation.get_asm(print_generic_op_form=True))
   subprocess.run(['build/gqa-opt',str(input_file),'--allow-unregistered-dialect','--attention-guard-online-reductions','-o',str(output_file)],check=True)
   changed=ir.Module.parse(output_file.read_text())
   updates=[x for x in walk(changed.operation) if x.name=='iree_linalg_ext.online_attention']
   assert len(updates)==1
   if str(op.attributes['lowering_config'])!=str(updates[0].attributes['lowering_config']):
    op.attributes['lowering_config']=updates[0].attributes['lowering_config'];count+=1
  Path(destination).write_text(str(module));return count
if __name__=='__main__':print('guarded',guard(sys.argv[1],sys.argv[2],sys.argv[3]))
