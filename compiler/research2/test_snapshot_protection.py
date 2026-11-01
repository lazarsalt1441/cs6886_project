import os,sys,subprocess,unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];os.chdir(ROOT);sys.path.insert(0,str(Path(__file__).resolve().parent))
from study_cache import ir
from test_update_coalescing import fixture
os.environ['LD_LIBRARY_PATH']=str(ROOT/'toolchain/usr/lib/llvm-19/lib')+':'+str(ROOT/'toolchain/usr/lib/x86_64-linux-gnu')+':'+os.environ.get('LD_LIBRARY_PATH','')
NATIVE=Path('research2/results/cache-snapshot-pre-cow.mlir').read_text()
def opt(text,phase=False):
 with ir.Context():text=ir.Module.parse(text).operation.get_asm(print_generic_op_form=True,use_local_scope=True)
 flag='--attention-materialize-borrowed-cache-cow' if phase else '--attention-protect-cache-snapshots'
 out=subprocess.run(['build/gqa-opt','--allow-unregistered-dialect',flag],input=text,text=True,capture_output=True,check=True).stdout
 with ir.Context():ir.Module.parse(out).operation.verify()
 return out
class SnapshotProtectionTests(unittest.TestCase):
 def test_returned_updated_input(self):self.assertEqual(opt(fixture()).count('cache_snapshot_guard'),1)
 def test_single_use_no_snapshot(self):self.assertNotIn('cache_snapshot_guard',opt(fixture(single=True)))
 def test_idempotent_guard(self):
  out=opt(fixture());self.assertEqual(out,opt(out))
 def test_supported_types_and_ranks(self):
  for rank in range(1,6):
   for dtype in ['f16','bf16','i32']:self.assertEqual(opt(fixture(rank=rank,dtype=dtype)).count('cache_snapshot_guard'),1)
 def test_read_only_identity_no_guard(self):
  self.assertNotIn('cache_snapshot_guard',opt('module {func.func @main(%x:tensor<64xf32>)->tensor<64xf32>{return %x:tensor<64xf32>}}'))
 def test_duplicate_snapshot_reuses_one_guard(self):
  text=fixture().replace(')->(tensor<64x4x4xf32>,tensor<64x4x4xf32>)',' )->(tensor<64x4x4xf32>,tensor<64x4x4xf32>,tensor<64x4x4xf32>)').replace('return %outer,%base : tensor<64x4x4xf32>,tensor<64x4x4xf32>','return %outer,%base,%base : tensor<64x4x4xf32>,tensor<64x4x4xf32>,tensor<64x4x4xf32>');self.assertEqual(opt(text).count('cache_snapshot_guard'),1)
 def test_phase_borrowed_import_clone(self):self.assertEqual(opt(NATIVE,True).count('borrowed_cache_cow'),1)
 def test_phase_idempotence(self):
  out=opt(NATIVE,True);self.assertEqual(out,opt(out,True))
 def test_phase_consuming_import_unchanged(self):
  text=NATIVE.replace('stream.tensor.import on(#hal.device.affinity<@__device_0>) %arg0','stream.tensor.import on(#hal.device.affinity<@__device_0>) consume %arg0');self.assertNotIn('borrowed_cache_cow',opt(text,True))
 def test_phase_affinity_preserved(self):
  out=opt(NATIVE,True)
  with ir.Context():
   m=ir.Module.parse(out);clones=[]
   def walk(op):
    if op.operation.name=='stream.async.clone':clones.append(op)
    for region in op.regions:
     for block in region.blocks:
      for child in block.operations:walk(child)
   walk(m.operation);self.assertEqual(len(clones),1);self.assertIn('affinity',clones[0].operation.get_asm(print_generic_op_form=True))
if __name__=='__main__':unittest.main(verbosity=2)
