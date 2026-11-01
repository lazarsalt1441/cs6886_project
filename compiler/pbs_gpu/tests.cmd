#!/bin/bash
#PBS -N attention_gpu
#PBS -q gpuq
#PBS -l select=1:ncpus=1:ngpus=1
#PBS -l walltime=08:00:00
#PBS -j oe
set -uo pipefail
cd "$PBS_O_WORKDIR"
# Add the site's Python/CUDA module commands here if needed.
source gpu_env/bin/activate || exit 1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export PYTHONUNBUFFERED=1
jobtag=${PBS_JOBID%%.*}
out="$PBS_O_WORKDIR/pbs_results/job_${jobtag}"
mkdir -p "$out"
finish() {
  # Copy partial results even when a command fails. SIGKILL/walltime may prevent this.
  python pbs_gpu/collect.py "$out" || true
  tar -czf "$PBS_O_WORKDIR/gpu_results_${jobtag}.tar.gz" -C "$PBS_O_WORKDIR/pbs_results" "job_${jobtag}" || true
}
trap finish EXIT
nvidia-smi -q > "$out/nvidia-smi.txt" 2>&1
iree-compile --version > "$out/compiler.txt" 2>&1
python -m pip freeze > "$out/packages.txt"
uname -a > "$out/system.txt"
arch=${GPU_ARCH:-$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1 | tr -d '.[:space:]' | sed 's/^/sm_/')}
if [[ ! "$arch" =~ ^sm_[0-9]+[a-z]?$ ]]; then echo 'Cannot detect GPU arch; set GPU_ARCH through qsub -v'; exit 2; fi
printf '%s\n' "$arch" > "$out/arch.txt"
family=${FAMILY:-all}
trials=${TRIALS:-101}
seeds=${SEEDS:-10}
step=0
run() { step=$((step+1)); echo "Running $*"; "$@" > "$out/step_${step}.log" 2>&1; rc=$?; printf '%s\t%s\n' "$rc" "$*" >> "$out/commands.tsv"; }
# Each family may be submitted separately, but do not run concurrent jobs in the same directory.
if [[ "$family" == all || "$family" == cache ]]; then
 run python research2/gpu_suite.py --arch "$arch" --family attention --trials "$trials"
 run python research2/gpu_suite.py --arch "$arch" --family block --trials "$trials"
fi
if [[ "$family" == all || "$family" == winner ]]; then
 run python research2/gpu_suite.py --arch "$arch" --family winner --trials "$trials"
fi
if [[ "$family" == all || "$family" == chain ]]; then
 if [[ "$arch" == sm_80 ]]; then extra=--prepared-ir; else extra=--skip-root-cow; fi
 run python research2/extension_gpu_suite.py --arch "$arch" "$extra" --trials "$trials"
fi
if [[ "$family" == all || "$family" == projection ]]; then
 run python research/gpu_research_suite.py --arch "$arch" --full --trials "$trials"
fi
run python pbs_gpu/validate.py --arch "$arch" --family "$family" --seeds "$seeds" --output "$out/validation.json"
cp "$out/commands.tsv" "$out/command-status.txt"
# The validation report, not the mere existence of an archive, determines correctness.
