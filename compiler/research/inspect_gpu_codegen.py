"""Static code inventory, not measured traffic or dynamic instruction counts."""
from pathlib import Path
import re,json
rows=[]
for folder in sorted(Path('research/results/gpu/sm_80').iterdir()):
 row={'case':folder.name,'variants':{}}
 for variant in ['baseline','optimized']:
  code=folder/(variant+'-codegen');ptx=list(code.glob('*.ptx'));text='\n'.join(p.read_text() for p in ptx)
  row['variants'][variant]={'ptx_bytes':sum(p.stat().st_size for p in ptx),'kernel_entries':len(re.findall(r'\.entry\s+',text)),'static_ld_global_sites':len(re.findall(r'\bld\.global(?:\.[\w]+)*\s',text)),'static_st_global_sites':len(re.findall(r'\bst\.global(?:\.[\w]+)*\s',text)),'llvm_bytes':sum(p.stat().st_size for p in code.glob('*.optimized.ll'))}
 rows.append(row)
Path('research/results/gpu-static-codegen.json').write_text(json.dumps({'warning':'Static instruction sites are not runtime traffic or dynamic instruction counts. Compare launch dimensions and profile on hardware.','rows':rows},indent=2))
for r in rows:
 if 'h8_m1_n512_l128_d64_v64_s1' in r['case'] or 'h16_m1_n1024' in r['case']:print(r)
