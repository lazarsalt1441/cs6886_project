"""Structural contracts for demand-driven cache-version read forwarding."""
import os,subprocess,unittest,re
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];os.chdir(ROOT)
os.environ['LD_LIBRARY_PATH']=str(ROOT/'toolchain/usr/lib/llvm-19/lib')+':'+str(ROOT/'toolchain/usr/lib/x86_64-linux-gnu')+':'+os.environ.get('LD_LIBRARY_PATH','')
from iree.compiler import ir
BASE=Path('research2/results/forks-probe/generic.mlir').read_text()
def opt(text):return subprocess.run(['build/gqa-opt','--allow-unregistered-dialect','--attention-forward-cache-reads','--canonicalize'],input=text,text=True,capture_output=True,check=True).stdout
def generic(text):
 with ir.Context():return ir.Module.parse(text).operation.get_asm(print_generic_op_form=True)
class CacheTests(unittest.TestCase):
 def test_forked_versions_forwarded(self):
  out=opt(BASE);self.assertEqual(out.count('attention.cache_read_forwarded'),4);self.assertNotIn('tensor.insert_slice',out)
 def test_single_version_preserved(self):
  text=generic(Path('research2/results/probe/attention-only/baseline.mlir').read_text());out=opt(text)
  self.assertNotIn('attention.cache_read_forwarded',out);self.assertIn('tensor.insert_slice',out)
 def test_force_single_version(self):
  text=Path('research2/results/probe/attention-only/baseline.mlir').read_text().replace(' -> tensor<1x8x1x32xf32> {',' -> tensor<1x8x1x32xf32> attributes {attention.force_cache_forwarding} {',1)
  self.assertEqual(opt(generic(text)).count('attention.cache_read_forwarded'),2)
 def test_dense_demand_preserved(self):self.assertNotIn('attention.cache_read_forwarded',opt(BASE.replace('4096x2x32','128x2x32')))
 def test_idempotent(self):
  first=opt(BASE);self.assertEqual(first,opt(first))
 def test_positive_strides(self):
  text=Path('research2/results/forks-probe/baseline.mlir').read_text().replace('[1, 1, 1] : tensor<2x2x32xf32>','[2, 1, 1] : tensor<2x2x32xf32>');out=opt(generic(text))
  self.assertEqual(out.count('attention.cache_read_forwarded'),4);self.assertIn('arith.remui',out);self.assertIn('arith.divui',out)
 def test_rank_reduced_write_rejected(self):
  text=Path('research2/results/probe/attention-only/baseline.mlir').read_text().replace('tensor<2x2x32xf32>','tensor<2x32xf32>').replace('[2, 2, 32] [1, 1, 1]','[2, 1, 32] [1, 1, 1]')
  self.assertNotIn('attention.cache_read_forwarded',opt(generic(text)))
 def test_returned_version_retained(self):
  with ir.Context(),ir.Location.unknown():
   mod=ir.Module.parse(BASE);f=mod.body.operations[0];block=f.regions[0].blocks[0];write=next(o for o in block.operations if o.operation.name=='tensor.insert_slice');ret=list(block.operations)[-1]
   old=ir.FunctionType(ir.TypeAttr(f.attributes['function_type']).value);f.attributes['function_type']=ir.TypeAttr.get(ir.FunctionType.get(list(old.inputs),list(old.results)+[write.results[0].type]))
   values=list(ret.operands)+[write.results[0]]
   with ir.InsertionPoint(ret):ir.Operation.create('func.return',operands=values)
   ret.operation.erase();text=mod.operation.get_asm(print_generic_op_form=True)
  out=opt(text);self.assertEqual(out.count('attention.cache_read_forwarded'),3);self.assertIn('tensor.insert_slice',out)
 def test_f16_copy_supported(self):self.assertEqual(opt(BASE.replace('f32','f16')).count('attention.cache_read_forwarded'),4)
 def test_bf16_copy_supported(self):self.assertEqual(opt(BASE.replace('f32','bf16')).count('attention.cache_read_forwarded'),4)
 def test_dynamic_offset_supported(self):
  text=Path('research2/results/forks-probe/baseline.mlir').read_text();text=text.replace('%arg5: tensor<128xi64>)','%arg5: tensor<128xi64>, %pos: tensor<i64>)',1)
  text=text.replace('    %cst =','    %position = tensor.extract %pos[] : tensor<i64>\n    %slot = arith.index_cast %position : i64 to index\n    %cst =',1).replace('[37, 0, 0]','[%slot, 0, 0]')
  self.assertEqual(opt(generic(text)).count('attention.cache_read_forwarded'),4)
 def test_conditional_not_eager_select(self):
  out=opt(BASE);self.assertEqual(out.count('scf.if'),4);self.assertNotIn('arith.select',out)
if __name__=='__main__':unittest.main(verbosity=2)
