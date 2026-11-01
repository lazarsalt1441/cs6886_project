"""Emulate observed PTX range reduction; not a GPU execution measurement."""
import json,sys,struct
from pathlib import Path
import numpy as np
import torch
F=np.float32
def fma(a,b,c):return F(float(a)*float(b)+float(c))
def approx(x,cos=False):
 q=F(np.floor(F(x*F(2/np.pi))));r=F(x+F(q*F(-np.pi/2)));n=int(q);r2=F(r*r)
 # PTX selects sin/cos polynomial by quadrant, then sign.
 use_cos=(n%2==0) if cos else (n%2==1)
 coeff=[struct.unpack('!f',bytes.fromhex(h))[0] for h in (['BF000000','3D2AAAA4','BAB60981','37CFAB9C','B48B634D'] if use_cos else ['BE2AAAAB','3C088898','B9501096','363938A8','B2D70013'])]
 p=coeff[-1]
 for c in coeff[-2::-1]:p=fma(r2,p,c)
 p=fma(r2,p,F(1));v=F((F(1) if use_cos else r)*p)
 negative=(n%4 in [1,2]) if cos else (n%4 in [2,3])
 return F(-v if negative else v)
def stable(x,cos=False):
 d=float(x);r=F(d-np.floor(d/(2*np.pi)+.5)*(2*np.pi))
 return approx(r,cos)
inv=1/(100000**(torch.arange(0,64,2).float()/64));rows=[]
for n in [128,512,2048,3072,4096,8191]:
 x=(inv*n).numpy()
 for cos in [False,True]:
  ref=np.cos(x.astype('float64')) if cos else np.sin(x.astype('float64'))
  a=np.array([approx(z,cos) for z in x]);b=np.array([stable(z,cos) for z in x]);rows.append(dict(position=n,op='cos' if cos else 'sin',observed_ptx_style_max_abs=float(abs(a-ref).max()),compensated_max_abs=float(abs(b-ref).max())))
out=Path(__file__).with_name('trig_evidence.json');out.write_text(json.dumps(rows,indent=2));print(json.dumps(rows[-6:],indent=2))
