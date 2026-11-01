import os,sys,subprocess,unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];os.chdir(ROOT);sys.path.insert(0,str(Path(__file__).resolve().parent))
from study_cache import ir
os.environ['LD_LIBRARY_PATH']=str(ROOT/'toolchain/usr/lib/llvm-19/lib')+':'+str(ROOT/'toolchain/usr/lib/x86_64-linux-gnu')+':'+os.environ.get('LD_LIBRARY_PATH','')
def fixture(rank=3,pool=64,dtype='f32',second=9,stride=1,observed=False,single=False,different_tail=False):
 shape=[pool]+[4]*(rank-1);small=[2]+[2]*(rank-1);ty=lambda x:'tensor<'+'x'.join(map(str,x))+'x'+dtype+'>';large,update=ty(shape),ty(small);resulttypes=[large]+([] if single else [large])+([large] if observed else []);returned=['%outer']+([] if single else ['%base'])+(['%inner'] if observed else []);tail=(',1' if different_tail and rank>1 else ',0')+',0'*(rank-2) if rank>1 else ''
 return f'''module {{func.func @main(%base:{large},%a:{update},%b:{update})->({','.join(resulttypes)}) {{
 %inner=tensor.insert_slice %a into %base[7{',0'*(rank-1)}] {small} {[1]*rank} : {update} into {large}
 %outer=tensor.insert_slice %b into %inner[{second}{tail}] {small} {[stride]+[1]*(rank-1)} : {update} into {large}
 return {','.join(returned)} : {','.join(resulttypes)}
 }} }}'''
def opt(text):
 with ir.Context():text=ir.Module.parse(text).operation.get_asm(print_generic_op_form=True)
 out=subprocess.run(['build/gqa-opt','--allow-unregistered-dialect','--attention-coalesce-cache-updates','--canonicalize'],input=text,text=True,capture_output=True,check=True).stdout
 with ir.Context():ir.Module.parse(out).operation.verify()
 return out
class CoalescingTests(unittest.TestCase):
 def test_adjacent(self):
  out=opt(fixture());self.assertEqual(out.count('tensor.insert_slice'),1);self.assertIn('tensor.concat',out);self.assertIn('cache_updates_coalesced',out)
 def test_ranks1_to5(self):
  for r in range(1,6):self.assertIn('cache_updates_coalesced',opt(fixture(rank=r)))
 def test_prepend(self):self.assertIn('cache_updates_coalesced',opt(fixture(second=5)))
 def test_gap_rejected(self):self.assertNotIn('cache_updates_coalesced',opt(fixture(second=10)))
 def test_overlap_rejected(self):self.assertNotIn('cache_updates_coalesced',opt(fixture(second=8)))
 def test_stride_rejected(self):self.assertNotIn('cache_updates_coalesced',opt(fixture(stride=2)))
 def test_different_tail_region_rejected(self):self.assertNotIn('cache_updates_coalesced',opt(fixture(different_tail=True)))
 def test_observed_intermediate_rejected(self):self.assertNotIn('cache_updates_coalesced',opt(fixture(observed=True)))
 def test_single_use_pool_profitability_rejected(self):self.assertNotIn('cache_updates_coalesced',opt(fixture(single=True)))
 def test_dense_update_profitability_rejected(self):self.assertNotIn('cache_updates_coalesced',opt(fixture(pool=16,second=11).replace('[2, 2, 2]','[3, 2, 2]').replace('tensor<2x2x2xf32>','tensor<3x2x2xf32>').replace('[7,','[8,')))
 def test_float_and_integer_types(self):
  for d in ['f16','bf16','i32']:self.assertIn('cache_updates_coalesced',opt(fixture(dtype=d)))
 def test_long_chain_concat_flattened(self):
  text=fixture().replace('return %outer,%base','%third=tensor.insert_slice %a into %outer[11,0,0] [2,2,2] [1,1,1] : tensor<2x2x2xf32> into tensor<64x4x4xf32>\n return %third,%base')
  out=opt(text);self.assertEqual(out.count('tensor.insert_slice'),1);self.assertEqual(out.count('tensor.concat'),1);self.assertIn('tensor<6x2x2xf32>',out)
 def test_idempotence(self):
  out=opt(fixture());self.assertEqual(out,opt(out))
if __name__=='__main__':unittest.main(verbosity=2)
