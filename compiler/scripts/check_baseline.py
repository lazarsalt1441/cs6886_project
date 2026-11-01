import sys
sys.path.insert(0,'scripts')
import numpy as np
import iree.runtime as rt
from export_gqa import GQA,inputs
x=inputs(); expected=GQA()(*x).numpy()
for variant in ['baseline','optimized']:
    config=rt.Config('local-sync')
    vm=rt.VmModule.mmap(config.vm_instance,f'results/{variant}.vmfb')
    context=rt.SystemContext(config=config); context.add_vm_module(vm)
    y=context.modules.module.main(*[t.numpy() for t in x]).to_host()
    print(variant,'shape',y.shape,'max_abs',np.max(np.abs(y-expected)))
    np.testing.assert_allclose(y,expected,rtol=1e-4,atol=1e-5)
