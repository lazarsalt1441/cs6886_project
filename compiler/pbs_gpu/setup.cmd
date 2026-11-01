#!/bin/bash
#PBS -N attention_setup
#PBS -q gpuq
#PBS -l select=1:ncpus=1:ngpus=1
#PBS -l walltime=02:00:00
#PBS -j oe
set -euo pipefail
cd "$PBS_O_WORKDIR"
# Add the site's Python/CUDA module commands here if needed.
python3.12 -m venv gpu_env
source gpu_env/bin/activate
python -m pip install --upgrade pip
python -m pip install torch==2.14.1 --index-url "${TORCH_INDEX:-https://download.pytorch.org/whl/cpu}"
python -m pip install iree-base-compiler==3.13.0 iree-base-runtime==3.13.0 iree-turbine==3.9.0 numpy==2.3.5 ml_dtypes==0.6.0
python -m pip freeze > gpu_environment.txt
nvidia-smi > gpu_setup_hardware.txt
python -c "import iree.runtime as r; assert r.get_driver('cuda').query_available_devices(); print('CUDA driver ready')"
