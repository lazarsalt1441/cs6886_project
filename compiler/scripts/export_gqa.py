"""Deterministic, explicit-repeat GQA reference and actual Turbine export."""
import argparse
from pathlib import Path
import torch
from iree.turbine import aot

class GQA(torch.nn.Module):
    def forward(self, q, k, v):
        ratio = q.shape[1] // k.shape[1]
        k = k.repeat_interleave(ratio, dim=1)
        v = v.repeat_interleave(ratio, dim=1)
        p = torch.softmax((q @ k.transpose(-2, -1)) * q.shape[-1] ** -0.5, dim=-1)
        return p @ v

def inputs(b=1,hq=8,hkv=2,m=1,n=128,d=32):
    torch.manual_seed(42)
    return (torch.randn(b,hq,m,d),torch.randn(b,hkv,n,d),torch.randn(b,hkv,n,d))

if __name__ == '__main__':
    p=argparse.ArgumentParser(); p.add_argument('--output',default='ir/torch.mlir')
    a=p.parse_args(); x=inputs(); model=GQA().eval()
    ep=torch.export.export(model,x)
    Path('ir/torch-export.txt').write_text(str(ep))
    out=aot.export(ep); out.save_mlir(a.output)
    print('Exported',a.output, 'shape', list(model(*x).shape))
