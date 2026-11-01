import os,sys,subprocess,unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];os.chdir(ROOT);sys.path.insert(0,str(Path(__file__).resolve().parent))
from study_cache import ir
os.environ['LD_LIBRARY_PATH']=str(ROOT/'toolchain/usr/lib/llvm-19/lib')+':'+str(ROOT/'toolchain/usr/lib/x86_64-linux-gnu')+':'+os.environ.get('LD_LIBRARY_PATH','')
def fixture(rank=3,dtype='f32',different_base=False,different_offset=False,dynamic=False,strided=False,elementwise=False):
 shape=[64]+[2]*(rank-1);small=[2]+[2]*(rank-1);ty=lambda x:'tensor<'+'x'.join(map(str,x))+'x'+dtype+'>';large,update=ty(shape),ty(small);dom=','.join('d'+str(i) for i in range(rank));cond='tensor<'+'x'.join(map(str,shape))+'xi1>' if elementwise else 'tensor<i1>';identity=f'affine_map<({dom})->({dom})>';cm=identity if elementwise else f'affine_map<({dom})->()>';off='%p' if dynamic else '7';offb='8' if different_offset else off;steps=[2]+[1]*(rank-1) if strided else [1]*rank
 extra=',%p:index' if dynamic else ''
 return f'''module {{func.func @main(%base:{large},%other:{large},%a:{update},%b:{update},%c:{cond}{extra})->{large} {{
 %wa=tensor.insert_slice %a into %base[{off}{',0'*(rank-1)}] {small} {steps} : {update} into {large}
 %wb=tensor.insert_slice %b into {'%other' if different_base else '%base'}[{offb}{',0'*(rank-1)}] {small} {steps} : {update} into {large}
 %e=tensor.empty():{large}
 %r=linalg.generic {{indexing_maps=[{cm},{identity},{identity},{identity}],iterator_types=[{','.join('"parallel"' for _ in range(rank))}]}} ins(%c,%wa,%wb:{cond},{large},{large}) outs(%e:{large}) {{
 ^bb0(%c0:i1,%a0:{dtype},%b0:{dtype},%out:{dtype}):
 %s=arith.select %c0,%a0,%b0:{dtype}
 linalg.yield %s:{dtype}
 }} ->{large}
 return %r:{large}
 }} }}'''
def opt(text):
 with ir.Context():text=ir.Module.parse(text).operation.get_asm(print_generic_op_form=True)
 out=subprocess.run(['build/gqa-opt','--allow-unregistered-dialect','--attention-factor-cache-commit','--canonicalize'],input=text,text=True,capture_output=True,check=True).stdout
 with ir.Context():ir.Module.parse(out).operation.verify()
 return out
class CommitFactoringTests(unittest.TestCase):
 def test_same_region(self):
  out=opt(fixture());self.assertEqual(out.count('commit_factored'),1);self.assertEqual(out.count('tensor.insert_slice'),1)
 def test_all_supported_ranks(self):
  for rank in range(1,6):self.assertIn('commit_factored',opt(fixture(rank=rank)))
 def test_dynamic_shared_offset(self):self.assertIn('commit_factored',opt(fixture(dynamic=True)))
 def test_strided_shared_region(self):self.assertIn('commit_factored',opt(fixture(strided=True)))
 def test_f16(self):self.assertIn('commit_factored',opt(fixture(dtype='f16')))
 def test_bf16(self):self.assertIn('commit_factored',opt(fixture(dtype='bf16')))
 def test_integer(self):self.assertIn('commit_factored',opt(fixture(dtype='i32')))
 def test_different_base_rejected(self):self.assertNotIn('commit_factored',opt(fixture(different_base=True)))
 def test_different_offset_rejected(self):self.assertNotIn('commit_factored',opt(fixture(different_offset=True)))
 def test_elementwise_condition_rejected(self):self.assertNotIn('commit_factored',opt(fixture(elementwise=True)))
 def test_idempotent(self):
  out=opt(fixture());self.assertEqual(out,opt(out))
 def test_original_observed_versions_retained(self):
  text=fixture().replace(')->tensor<64x2x2xf32> {',')->(tensor<64x2x2xf32>,tensor<64x2x2xf32>) {').replace('return %r:tensor<64x2x2xf32>','return %r,%wa:tensor<64x2x2xf32>,tensor<64x2x2xf32>');self.assertEqual(opt(text).count('tensor.insert_slice'),2)
if __name__=='__main__':unittest.main(verbosity=2)
