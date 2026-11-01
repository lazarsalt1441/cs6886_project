import sys,os,subprocess,unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];os.chdir(ROOT);sys.path.insert(0,str(Path(__file__).resolve().parent))
from study_cache import ir
os.environ['LD_LIBRARY_PATH']=str(ROOT/'toolchain/usr/lib/llvm-19/lib')+':'+str(ROOT/'toolchain/usr/lib/x86_64-linux-gnu')+':'+os.environ.get('LD_LIBRARY_PATH','')
def fixture(indexshape='16x1',outputshape='16x4',integer='i64',depth=1):
 source='257x4';outtype=f'tensor<{outputshape}xf32>';idxtype=f'tensor<{indexshape}x{integer}>'
 dynamic = (' %c0 = arith.constant 0 : index\n %n = tensor.dim %idx, %c0 : '+idxtype+'\n') if '?' in indexshape else ''
 emptyargs='%n' if '?' in outputshape else ''
 return f'''module {{ func.func @main(%src: tensor<{source}xf32>, %idx: {idxtype}) -> {outtype} {{
 {dynamic}%empty = tensor.empty({emptyargs}) : {outtype}
 %r = iree_linalg_ext.gather dimension_map = [{', '.join(str(i) for i in range(depth))}] ins(%src, %idx : tensor<{source}xf32>, {idxtype}) outs(%empty : {outtype}) -> {outtype}
 return %r : {outtype}
 }} }}'''
def opt(text):
 with ir.Context():m=ir.Module.parse(text);text=m.operation.get_asm(print_generic_op_form=True)
 return subprocess.run(['build/gqa-opt','--allow-unregistered-dialect','--attention-canonicalize-gather-indices','--canonicalize'],input=text,text=True,capture_output=True,check=True).stdout
class GatherProjectionTests(unittest.TestCase):
 def test_explicit_component_removed(self):self.assertIn('tensor<16xi64>',opt(fixture()))
 def test_implicit_component_unchanged(self):self.assertNotIn('index_projection_normalized',opt(fixture('16')))
 def test_two_batch_axes(self):
  s=opt(fixture('2x8x1','2x8x4'));self.assertIn('tensor<2x8xi64>',s);self.assertIn('[[0], [1, 2]]',s)
 def test_i32(self):self.assertIn('tensor<16xi32>',opt(fixture(integer='i32')))
 def test_index(self):self.assertIn('tensor<16xindex>',opt(fixture(integer='index')))
 def test_dynamic_batch(self):self.assertIn('tensor<?xi64>',opt(fixture('?x1','?x4')))
 def test_mask_is_preserved(self):
  text=fixture().replace('%idx: tensor<16x1xi64>)','%idx: tensor<16x1xi64>, %mask: tensor<16xi1>)').replace('ins(%src, %idx : tensor<257x4xf32>, tensor<16x1xi64>)','ins(%src, %idx, %mask : tensor<257x4xf32>, tensor<16x1xi64>, tensor<16xi1>)')
  out=opt(text)
  with ir.Context():ir.Module.parse(out).operation.verify()
  self.assertIn('tensor<16xi1>',out);self.assertIn('tensor<16xi64>',out)
 def test_depth_two_unchanged(self):self.assertNotIn('index_projection_normalized',opt(fixture('16x2','16',depth=2)))
 def test_idempotent(self):
  s=opt(fixture());self.assertEqual(s,opt(s))
if __name__=='__main__':unittest.main(verbosity=2)
