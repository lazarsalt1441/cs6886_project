import os,sys,subprocess,unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];os.chdir(ROOT);sys.path.insert(0,str(Path(__file__).resolve().parent))
from study_cache import ir
os.environ['LD_LIBRARY_PATH']=str(ROOT/'toolchain/usr/lib/llvm-19/lib')+':'+str(ROOT/'toolchain/usr/lib/x86_64-linux-gnu')+':'+os.environ.get('LD_LIBRARY_PATH','')
BASE=Path('research2/results/study/p4096_n128_h2_g4_d32_m1_f2_o37_t2/lifted-normalized.mlir').read_text()
def opt(text,ratio=4):
 with ir.Context():text=ir.Module.parse(text).operation.get_asm(print_generic_op_form=True)
 out=subprocess.run(['build/gqa-opt','--allow-unregistered-dialect','--attention-forward-gather-updates=min-pool-demand-ratio='+str(ratio),'--canonicalize'],input=text,text=True,capture_output=True,check=True).stdout
 with ir.Context():ir.Module.parse(out).operation.verify()
 return out
class GatherForwardingTests(unittest.TestCase):
 def test_forks(self):
  text=opt(BASE);self.assertEqual(text.count('cache_gather_forwarded'),4);self.assertNotIn('tensor.insert_slice',text)
 def test_preserves_registered_gather_properties(self):
  out=opt(BASE);self.assertEqual(out.count('dimension_map = array<i64: 0>'),8)
 def test_idempotent(self):
  out=opt(BASE);self.assertEqual(out,opt(out))
 def test_single_version_control(self):
  text=Path('research2/results/study/p4096_n128_h2_g4_d32_m1_f1_o37_t2/lifted-normalized.mlir').read_text();self.assertNotIn('cache_gather_forwarded',opt(text))
 def test_dense_control(self):
  text=Path('research2/results/study/p256_n256_h2_g2_d32_m1_f2_o37_t2/lifted-normalized.mlir').read_text();self.assertNotIn('cache_gather_forwarded',opt(text))
 def test_half_pool_profitability_rejected(self):
  text=Path('research2/results/study/p1024_n512_h2_g4_d32_m1_f2_o37_t2/lifted-normalized.mlir').read_text();self.assertNotIn('cache_gather_forwarded',opt(text));self.assertIn('cache_gather_forwarded',opt(text,ratio=2))
 def test_strided_rejected(self):
  text=BASE.replace('[1, 1, 1] : tensor<2x2x32xf32>','[2, 1, 1] : tensor<2x2x32xf32>');self.assertNotIn('cache_gather_forwarded',opt(text))
 def test_translation_is_integer_only(self):
  out=opt(BASE);self.assertIn('arith.subi',out);self.assertIn('arith.cmpi ult',out);self.assertIn('arith.select',out);self.assertNotIn('arith.mulf',out);self.assertNotIn('allow_reassociation',out)
if __name__=='__main__':unittest.main(verbosity=2)
