"""Retain CPU assembly plus scheduled-copy evidence, without DRAM claims."""
import sys,json,subprocess,hashlib,re
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
from study_cache import run,STRONG,copy_stats

def main():
 case='p65536_n128_h2_g4_d32_m1_f2_o37_t2';source=Path('research2/results/study')/case;root=Path('research2/results/cpu-codegen');root.mkdir(parents=True,exist_ok=True);rows={}
 for variant,stem in [('strong','baseline'),('patch-forwarded','patch-forwarded')]:
  w=root/variant;w.mkdir(parents=True,exist_ok=True)
  run(['iree-compile',source/(stem+'.mlir'),'--iree-hal-target-backends=llvm-cpu','--iree-llvmcpu-target-cpu=host',*STRONG,'--iree-hal-dump-executable-files-to='+str(w),'--iree-hal-dump-executable-intermediates-to='+str(w),'-o',w/'module.vmfb'],w/'compile.log')
  objects=list(w.glob('*.o'));disassembly=[]
  for obj in objects:
   output=subprocess.check_output(['objdump','-d',str(obj)],text=True);(obj.with_suffix('.assembly.txt')).write_text(output);disassembly.append(output)
  hal=(source/(variant+'-hal.mlir')).read_text();text=(source/(stem+'.mlir')).read_text()
  rows[variant]={'insert_slice_ops':text.count('tensor.insert_slice'),'scheduled_copy_bytes':copy_stats(hal),'scheduled_copy_bytes_sum':sum(copy_stats(hal)),'dispatches':hal.count('hal.command_buffer.dispatch'),'objects':len(objects),'assembly_contains_gather_mnemonic':bool(re.search(r'\bvgather', ''.join(disassembly))),'files':{str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in w.iterdir() if p.is_file()}}
 (root/'analysis.json').write_text(json.dumps({'case':case,'variants':rows,'interpretation':'HAL copy lengths are scheduled memory movement. Assembly and removed pool-sized copies support elimination of materialization; no hardware DRAM bytes or cache-miss measurement was available.'},indent=2));print({v:(r['insert_slice_ops'],r['scheduled_copy_bytes_sum'],r['objects']) for v,r in rows.items()},flush=True)
if __name__=='__main__':main()
