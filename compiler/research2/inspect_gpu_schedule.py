"""CUDA host schedule evidence; does not execute GPU or count DRAM traffic."""
import sys,json,re
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
from study_cache import run,STRONG,copy_stats

def main():
 case='p65536_n128_h2_g4_d32_m1_f2_o37_t2';source=Path('research2/results/study')/case;w=Path('research2/results/gpu-schedule');w.mkdir(parents=True,exist_ok=True);rows={}
 for v,stem in [('strong','baseline'),('lifted-normalized','lifted-normalized'),('forwarded','forwarded'),('patch-forwarded','patch-forwarded')]:
  path=w/(v+'.mlir');run(['iree-compile',source/(stem+'.mlir'),'--iree-hal-target-backends=cuda','--iree-cuda-target=sm_80',*STRONG,'--compile-to=hal','-o',path],w/(v+'.log'));text=path.read_text();copies=copy_stats(text)
  constants={k:int(v) for k,v in re.findall(r'(%[\w]+) = arith.constant (\d+) : index',text)};allocs=[]
  for line in text.splitlines():
   if 'hal.allocator.allocate' in line:
    match=re.search(r'size\((%[\w]+)\)',line)
    if match and match[1] in constants:allocs.append(constants[match[1]])
  rows[v]={'scheduled_copy_bytes':copies,'scheduled_copy_bytes_sum':sum(copies),'static_allocation_sizes':allocs,'dispatches':text.count('hal.command_buffer.dispatch')}
 Path('research2/results/gpu-schedule.json').write_text(json.dumps({'case':case,'arch':'sm_80','compile_only':True,'variants':rows,'note':'HAL scheduled copies and static allocation sizes only. No GPU execution, numerical validation, timing or DRAM counters.'},indent=2));print(rows,flush=True)
if __name__=='__main__':main()
