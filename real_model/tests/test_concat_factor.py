import os,subprocess,unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];TOOL=ROOT/'compiler/build/gqa-opt'
os.environ['LD_LIBRARY_PATH']=str(ROOT/'compiler/toolchain/usr/lib/llvm-19/lib')+':'+str(ROOT/'compiler/toolchain/usr/lib/x86_64-linux-gnu')
def source(history=8,patch=1,different=False,elementwise=False):
 h='?' if history is None else str(history);total='?' if history is None else str(history+patch)
 cond=f'tensor<1x2x{total}x4xi1>' if elementwise else 'tensor<i1>';condmap='(d0,d1,d2,d3)' if elementwise else '()'
 typ=f'tensor<1x2x{total}x4xf32>';ht=f'tensor<1x2x{h}x4xf32>';pt=f'tensor<1x2x{patch}x4xf32>'
 empty=('%dim = tensor.dim %history, %c2 : '+ht+'\n%size = arith.addi %dim, %cp : index\n%e = tensor.empty(%size) : '+typ) if history is None else '%e = tensor.empty() : '+typ
 return f'''module {{func.func @main(%history: {ht}, %other: {ht}, %a: {pt}, %b: {pt}, %c: {cond}) -> {typ} {{
 %c2 = arith.constant 2 : index
 %cp = arith.constant {patch} : index
 %x = tensor.concat dim(2) %history, %a : ({ht}, {pt}) -> {typ}
 %y = tensor.concat dim(2) {'%other' if different else '%history'}, %b : ({ht}, {pt}) -> {typ}
 {empty}
 %r = linalg.generic {{indexing_maps=[affine_map<(d0,d1,d2,d3)->{condmap}>, affine_map<(d0,d1,d2,d3)->(d0,d1,d2,d3)>, affine_map<(d0,d1,d2,d3)->(d0,d1,d2,d3)>, affine_map<(d0,d1,d2,d3)->(d0,d1,d2,d3)>], iterator_types=["parallel","parallel","parallel","parallel"]}}
 ins(%c,%x,%y : {cond},{typ},{typ}) outs(%e : {typ}) {{
 ^bb0(%c0: i1,%v: f32,%w: f32,%o: f32):
 %z = arith.select %c0,%v,%w : f32
 linalg.yield %z : f32
 }} -> {typ}
 return %r : {typ}
}}}}'''
def opt(s):return subprocess.run([TOOL,'--attention-factor-cache-commit','--canonicalize'],input=s,text=True,capture_output=True,check=True).stdout
class Tests(unittest.TestCase):
 def test_static(self):self.assertIn('attention.concat_commit_factored',opt(source()))
 def test_dynamic(self):self.assertIn('attention.concat_commit_factored',opt(source(None)))
 def test_different_history(self):self.assertNotIn('attention.concat_commit_factored',opt(source(different=True)))
 def test_elementwise(self):self.assertNotIn('attention.concat_commit_factored',opt(source(elementwise=True)))
 def test_large_patch(self):self.assertNotIn('attention.concat_commit_factored',opt(source(1,8)))
 def test_multirow(self):self.assertIn('attention.concat_commit_factored',opt(source(16,4)))
 def test_idempotent(self):
  a=opt(source());self.assertEqual(a,opt(a))
 def test_filecheck(self):
  subprocess.run([ROOT/'compiler/toolchain/usr/lib/llvm-19/bin/FileCheck',Path(__file__).with_name('concat.check')],input=opt(source()),text=True,check=True)
if __name__=='__main__':unittest.main(verbosity=2)
