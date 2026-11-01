"""Run genuine C++ pass, MLIR verifier, FileCheck, negative and multiuser tests."""
import os, subprocess, sys, unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];os.chdir(ROOT)
os.environ['LD_LIBRARY_PATH']=str(ROOT/'toolchain/usr/lib/llvm-19/lib')+':'+str(ROOT/'toolchain/usr/lib/x86_64-linux-gnu')+':'+os.environ.get('LD_LIBRARY_PATH','')
OPT=ROOT/'build/gqa-opt';FC=ROOT/'toolchain/usr/lib/llvm-19/bin/FileCheck'
SOURCE=(ROOT/'ir/linalg.mlir').read_text()
def optimize(s):
    return subprocess.run([OPT,'--gqa-head-reuse','--canonicalize'],input=s,text=True,capture_output=True,check=True).stdout
class Structural(unittest.TestCase):
    def test_exported_positive_filecheck(self):
        s=optimize(SOURCE)
        subprocess.run([FC,'tests/checks.txt','--check-prefix=REUSE'],input=s,text=True,check=True)
        self.assertNotIn('linalg.batch_matmul',s)
    def test_noncopy_broadcast_rejected(self):
        # Arithmetic in the producer must never be silently dropped.
        s=SOURCE.replace('linalg.yield %in : f32','%neg = arith.negf %in : f32\n      linalg.yield %neg : f32',2)
        out=optimize(s)
        subprocess.run([FC,'tests/checks.txt','--check-prefix=NEG'],input=out,text=True,check=True)
        self.assertEqual(out.count('arith.negf'),2)
    def test_wrong_source_yield_rejected(self):
        # A generic that yields its initialized destination is not a broadcast copy.
        s=SOURCE.replace('linalg.yield %in : f32','linalg.yield %out : f32',2)
        self.assertEqual(optimize(s).count('linalg.batch_matmul'),2)
    def test_wrong_axis_broadcast_rejected(self):
        # Swap head and replication axes where their extents happen to be equal.
        s=SOURCE.replace('1x2x4x128x32','1x2x2x128x32').replace('1x8x','1x4x').replace('tensor<8x','tensor<4x').replace('[1, 8,','[1, 4,')
        s=s.replace('-> (d0, d1, d3, d4)>','-> (d0, d2, d3, d4)>',1)
        self.assertEqual(optimize(s).count('linalg.batch_matmul'),2)
    def test_shared_producer_preserved(self):
        s=SOURCE.replace('-> tensor<1x8x1x32xf32> {','-> (tensor<1x8x1x32xf32>, tensor<1x2x4x128x32xf32>) {',1)
        s=s.replace('return %expanded_7 : tensor<1x8x1x32xf32>','return %expanded_7, %1 : tensor<1x8x1x32xf32>, tensor<1x2x4x128x32xf32>')
        out=optimize(s)
        subprocess.run([FC,'tests/checks.txt','--check-prefix=SHARED'],input=out,text=True,check=True)
        self.assertNotIn('linalg.batch_matmul',out)
        self.assertIn('tensor<1x2x4x128x32xf32>',out)
    def test_idempotent(self):
        a=optimize(SOURCE);self.assertEqual(a,optimize(a))
    def test_fp16_rejected(self):
        # All float literal constants are representable in fp16; use f16 source graph.
        s=SOURCE.replace('f32','f16').replace('0xFF800000','0xFC00')
        self.assertEqual(optimize(s).count('linalg.batch_matmul'),2)
    def test_dynamic_shapes_rejected(self):
        s=Path('tests/dynamic.mlir').read_text()
        out=optimize(s)
        self.assertEqual(out.count('linalg.batch_matmul'),1)
    def test_invalid_ir_fails_verification(self):
        s=SOURCE.replace('tensor<1x2x128x32xf32>','tensor<1x3x128x32xf32>')
        p=subprocess.run([OPT,'--gqa-head-reuse'],input=s,text=True,capture_output=True)
        self.assertNotEqual(p.returncode,0)
if __name__=='__main__':unittest.main(verbosity=2)
