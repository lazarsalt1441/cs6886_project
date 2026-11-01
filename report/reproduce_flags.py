"""Compile the saved original GQA fixture under the four flag combinations.

Run after installing requirements-cpu.txt and restoring artifacts:
  python report/reproduce_flags.py --output reproduced/flag_evidence
Counts are tied to this compiler's dispatch naming; inspect the saved Flow IR.
"""
import argparse, json, re, subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--compiler', default='iree-compile')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    source = ROOT / 'compiler/ir/linalg.mlir'
    if not source.exists():
        raise SystemExit('Run scripts/unpack_artifacts.py first')
    rows = []
    for generalize, reshape in [(False, False), (True, False), (False, True), (True, True)]:
        label = f'g{int(generalize)}_r{int(reshape)}'
        output = args.output / (label + '.mlir')
        cmd = [args.compiler, str(source), '--iree-hal-target-backends=llvm-cpu',
               '--iree-llvmcpu-target-cpu=host', '--iree-opt-level=O3', '--compile-to=flow',
               f'--iree-opt-generalize-matmul={str(generalize).lower()}',
               f'--iree-dispatch-creation-enable-aggressive-reshape-movement={str(reshape).lower()}',
               '-o', str(output)]
        if generalize and reshape:
            cmd += ['--mlir-print-ir-after-all', '--mlir-print-ir-after-change']
        with (args.output / (label + '.log')).open('w') as log:
            subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, check=True)
        calls = [line.strip() for line in output.read_text().splitlines() if 'flow.dispatch @' in line]
        broadcast = sum('elementwise_broadcast' in line for line in calls)
        rows.append(dict(generalize=generalize, aggressive_reshape=reshape,
                         broadcast_calls=broadcast, all_dispatch_calls=len(calls), commands=cmd))
        print(label, 'broadcast calls:', broadcast, 'all dispatch calls:', len(calls))
    (args.output / 'results.json').write_text(json.dumps(rows, indent=2) + '\n')

if __name__ == '__main__':
    main()
