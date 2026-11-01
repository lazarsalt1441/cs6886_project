"""Run all structural suites and the executable phase-local COW FileCheck."""
import os,sys,subprocess,json,re
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];os.chdir(ROOT);sys.path.insert(0,str(Path(__file__).resolve().parent))
from repair_snapshot_cow import repair
SUITES=['tests/test_structural.py','research/test_projection_pass.py','research/test_online_guard.py','research2/test_cache_forwarding.py','research2/test_gather_projection.py','research2/test_gather_forwarding.py','research2/test_commit_factoring.py','research2/test_update_coalescing.py','research2/test_snapshot_protection.py']
def main():
 rows=[]
 for source in SUITES:
  result=subprocess.run([sys.executable,source],capture_output=True,text=True);log=result.stdout+result.stderr;Path('research2/logs/final-'+Path(source).stem+'.log').write_text(log);match=re.search(r'Ran (\d+) tests?',log);row={'suite':source,'returncode':result.returncode,'tests':int(match[1]) if match else None};rows.append(row);assert result.returncode==0 and match,row
 source=Path('research2/patches/borrowed_cast_cow.mlir');dest=Path('research2/results/final-cow-filecheck.mlir');repair(source,dest,Path('research2/logs/final-cow-bridge.log'));result=subprocess.run(['iree-opt',str(dest),'--pass-pipeline=builtin.module(util.func(iree-stream-materialize-copy-on-write))'],capture_output=True,text=True,check=True);Path('research2/results/final-cow-pass.mlir').write_text(result.stdout);check=subprocess.run(['toolchain/usr/lib/llvm-19/bin/FileCheck',str(source),'--input-file=research2/results/final-cow-pass.mlir'],capture_output=True,text=True);Path('research2/logs/final-cow-filecheck.log').write_text(check.stdout+check.stderr);assert check.returncode==0,check.stderr
 total=sum(r['tests'] for r in rows);Path('research2/results/final-structural-tests.json').write_text(json.dumps({'suites':rows,'total_unittest_cases':total,'cow_filecheck_cases':2,'passed':True},indent=2));print(total,'structural unittest cases +2 COW FileCheck cases passed')
if __name__=='__main__':main()
