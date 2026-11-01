import os,subprocess,unittest,re
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];os.chdir(ROOT)
os.environ['LD_LIBRARY_PATH']=str(ROOT/'toolchain/usr/lib/llvm-19/lib')+':'+str(ROOT/'toolchain/usr/lib/x86_64-linux-gnu')+':'+os.environ.get('LD_LIBRARY_PATH','')
SOURCE=Path('research/ir/guard-bridge/input-0.mlir').read_text()
def guard(s):
 return subprocess.run(['build/gqa-opt','--allow-unregistered-dialect','--attention-guard-online-reductions'],input=s,text=True,capture_output=True,check=True).stdout
class OnlineGuardTests(unittest.TestCase):
 def test_forbids_feature_but_retains_token_tiling(self):
  s=guard(SOURCE);self.assertIn('vector_reduction = [0, 0, 8]',s)
  self.assertIn('attention.feature_reduction_guard',s)
 def test_idempotent(self):
  s=guard(SOURCE);self.assertEqual(s,guard(s))
 def test_already_legal_no_change(self):
  s=SOURCE.replace('vector_reduction = [0, 16, 16]','vector_reduction = [0, 0, 8]')
  self.assertNotIn('attention.feature_reduction_guard',guard(s))
 def test_unbounded_feature_not_vectorized(self):
  s=SOURCE.replace('256xf32','2048xf32')
  self.assertNotIn('attention.feature_reduction_guard',guard(s))
 def test_resource_aware_sequence_tile(self):
  s=SOURCE.replace('256xf32','512xf32')
  self.assertIn('vector_reduction = [0, 0, 4]',guard(s))
 def test_axis_permutation(self):
  def permute(match):
   text=match.group(2)
   for old,new in [('d0','TMP0'),('d1','TMP1'),('d2','TMP2')]:text=text.replace(old,new)
   for old,new in [('TMP0','d2'),('TMP1','d0'),('TMP2','d1')]:text=text.replace(old,new)
   return match.group(1)+text+match.group(3)
  s=re.sub(r'(affine_map<\(d0, d1, d2\) -> \()([^)]*)(\)>)',permute,SOURCE)
  s=s.replace('distribution = [32, 0, 0]','distribution = [0, 0, 32]').replace('vector_common_parallel = [16, 0, 0]','vector_common_parallel = [0, 0, 16]').replace('vector_reduction = [0, 16, 16]','vector_reduction = [16, 16, 0]')
  self.assertIn('vector_reduction = [0, 8, 0]',guard(s))
 def test_nonprojected_map_rejected(self):
  s=SOURCE.replace('-> (d2, d1)>','-> (d2, d1 + 1)>')
  self.assertNotIn('attention.feature_reduction_guard',guard(s))
 def test_inconsistent_map_rank_rejected(self):
  s=SOURCE.replace('#map1 = affine_map<(d0, d1, d2)', '#map1 = affine_map<(d0, d1, d2, d3)')
  self.assertNotIn('attention.feature_reduction_guard',guard(s))
 def test_actual_map_arity_rejected(self):
  s=SOURCE.replace('[#map, #map1, #map2, #map3, #map4, #map3, #map3]', '[#map, #map1, #map2, #map3, #map4, #map3]')
  self.assertNotIn('attention.feature_reduction_guard',guard(s))
 def test_nonidentity_score_region_rejected(self):
  s=SOURCE.replace('"iree_linalg_ext.yield"(%arg7)', '"iree_linalg_ext.yield"(%arg3)')
  self.assertNotIn('attention.feature_reduction_guard',guard(s))
 def test_wrong_map_arity_rejected(self):
  # A syntactically valid unrelated opaque operation must not be touched.
  s=SOURCE.replace('iree_linalg_ext.online_attention','other.online_attention')
  self.assertNotIn('attention.feature_reduction_guard',guard(s))
if __name__=='__main__':unittest.main(verbosity=2)
