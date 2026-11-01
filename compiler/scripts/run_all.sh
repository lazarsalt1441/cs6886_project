#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
source scripts/env.sh
mkdir -p logs
scripts/build.sh > logs/build.log 2>&1
python tests/test_structural.py > logs/structural.log 2>&1
python scripts/run_suite.py > logs/suite.log 2>&1
python scripts/extra_correctness.py > logs/extra-correctness.log 2>&1
python scripts/check_broadcast_frontend.py > logs/broadcast-frontend.log 2>&1
python scripts/evaluate_extensions.py > logs/extensions.log 2>&1
python scripts/measure_memory.py > logs/memory.log 2>&1
python scripts/rebenchmark.py > logs/pinned-benchmark.log 2>&1
python scripts/investigate_native_gqa.py > logs/native-gqa.log 2>&1
python scripts/compare_native.py > logs/native-comparison.log 2>&1
scripts/inspect_codegen.sh > logs/codegen.log 2>&1
python scripts/make_report.py > logs/report.log 2>&1
