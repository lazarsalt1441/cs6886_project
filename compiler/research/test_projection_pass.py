"""Contracts, profitability, shared uses and idempotence for projection placement."""
import os,sys,re,unittest,subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];os.chdir(ROOT)
os.environ['LD_LIBRARY_PATH']=str(ROOT/'toolchain/usr/lib/llvm-19/lib')+':'+str(ROOT/'toolchain/usr/lib/x86_64-linux-gnu')+':'+os.environ.get('LD_LIBRARY_PATH','')
BASE=Path('research/results/projections/h4_m1_n128_l64_d32_v32_s0_mask0')
NATIVE=Path('research/results/native-projections/h4_m1_n128_l64_d32_v32_s0')
ROTARY=Path('research/results/rotary-projections/h4_m1_n128_l64_d32_v32_r16')
PREFILL=Path('research/results/projections/h4_m128_n512_l128_d32_v32_s1_mask0')
def opt(text,passname='--attention-project-reassociate'):
 return subprocess.run(['build/gqa-opt','--allow-unregistered-dialect',passname,'--canonicalize'],input=text,text=True,capture_output=True,check=True).stdout
class ProjectionPassTests(unittest.TestCase):
 def test_no_contract_no_reassociation(self):
  self.assertNotIn('attention.projection_placement',opt((BASE/'baseline.mlir').read_text()))
 def test_decomposed_both(self):
  out=opt((BASE/'contract.mlir').read_text())
  self.assertIn('key-query',out);self.assertIn('value-output',out)
 def test_disable_key(self):
  text=(BASE/'contract.mlir').read_text().replace('attention.allow_reassociation','attention.disable_key_reassociation, attention.allow_reassociation')
  out=opt(text);self.assertNotIn('"key-query"',out);self.assertIn('value-output',out)
 def test_disable_value(self):
  text=(BASE/'contract.mlir').read_text().replace('attention.allow_reassociation','attention.disable_value_reassociation, attention.allow_reassociation')
  out=opt(text);self.assertIn('key-query',out);self.assertNotIn('value-output',out)
 def test_prefill_rejected_by_cost(self):
  self.assertNotIn('attention.projection_placement',opt((PREFILL/'contract.mlir').read_text()))
 def test_force_prefill_terminates(self):
  text=(PREFILL/'contract.mlir').read_text().replace('attention.allow_reassociation','attention.force_reassociation, attention.allow_reassociation')
  self.assertEqual(opt(text).count('attention.projection_placement'),2)
 def test_idempotent(self):
  for folder,file in [(BASE,'contract.mlir'),(NATIVE,'contract-generic.mlir'),(ROTARY,'contract.mlir')]:
   out=opt((folder/file).read_text());self.assertEqual(out,opt(out))
 def test_fp16_rejected(self):
  text=(BASE/'contract.mlir').read_text().replace('f32','f16').replace('0xFF800000','0xFC00')
  self.assertNotIn('attention.projection_placement',opt(text))
 def test_nonzero_seeds_rejected(self):
  text=(BASE/'contract.mlir').read_text().replace('arith.constant 0.000000e+00','arith.constant 1.000000e+00')
  self.assertNotIn('attention.projection_placement',opt(text))
 def test_shared_key_projection_not_duplicated(self):
  text=(BASE/'contract.mlir').read_text()
  name=re.search(r'(%[\w]+) = linalg.batch_matmul',text).group(1)
  text=text.replace('-> tensor<4x1x32xf32> attributes','-> (tensor<4x1x32xf32>, tensor<4x128x32xf32>) attributes',1)
  text=re.sub(r'return (%[\w]+) : tensor<4x1x32xf32>',lambda m:f'return {m.group(1)}, {name} : tensor<4x1x32xf32>, tensor<4x128x32xf32>',text)
  out=opt(text);self.assertNotIn('"key-query"',out);self.assertIn('value-output',out)
 def test_native_attention_retained_and_scaled(self):
  out=opt((NATIVE/'contract-generic.mlir').read_text())
  self.assertIn('"tm_tensor.attention"',out);self.assertIn('native-both',out)
  self.assertRegex(out,r'1\.41421')
 def test_rotary_shared_subspace(self):
  out=opt((ROTARY/'contract.mlir').read_text())
  self.assertIn('key-query-rotary-shared',out)
  self.assertIn('tensor<1x128x80xf32>',out)
 def test_rotary_factoring_ablation(self):
  text=(ROTARY/'contract.mlir').read_text().replace('attention.allow_reassociation','attention.disable_shared_concat, attention.allow_reassociation')
  out=opt(text);self.assertIn('tensor<4x128x80xf32>',out)
 def test_range_guard_nominal(self):
  text=Path('research/results/range-safety/nominal/safe-input.mlir').read_text()
  self.assertEqual(opt(text).count('attention.projection_placement'),2)
 def test_range_guard_overflow(self):
  text=Path('research/results/range-safety/overflow-counterexample/safe-input.mlir').read_text()
  self.assertNotIn('attention.projection_placement',opt(text))
 def test_range_guard_missing_bounds(self):
  text=(BASE/'contract.mlir').read_text().replace('attention.allow_reassociation','attention.require_range_safety, attention.allow_reassociation')
  self.assertNotIn('attention.projection_placement',opt(text))
if __name__=='__main__':unittest.main(verbosity=2)
