# Resume

Working project: /workspace/scratch/4088b657d6f5/real_llm_project.
Python: /workspace/scratch/4088b657d6f5/real_env/bin/python; prepend its bin to PATH; OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1.
Toolchain: Torch2.11.0+cpu, Python3.12, Transformers4.57.1, Turbine3.9.0, IREE compiler/runtime3.13.0; compiler89ece87b89852100c0a4ac3d515e14842d090967. Exact preparation environment in local_checks/preparation_environment.json. LLVM/MLIR19.1.1 SDK pinned by compiler/docs/toolchain.sha256; local executable compiler/build/gqa-opt (not bundled). Source compiler/src/CacheCommitFactoring.cpp plus ConcatCommit.inc, module-agnostic cache pass anchoring.
Pinned model HuggingFaceTB/SmolLM2-135M-Instruct revision12fd25f77366fa6b3b4b768ec3050bf629380bac; model download omitted from ZIP. Colab regenerates IRPA from checkpoint and exact wrapper names; no re-export/build required.

Successful commands, from project root:
python scripts/download_model.py
python compiler/scripts/fetch_mlir.py
bash compiler/scripts/build.sh
python scripts/prepare.py --set A
python scripts/prepare.py --set B
python tests/test_concat_factor.py
python tests/test_concat_numerical.py
python benchmark.py --set A --device cpu --contexts 8 32 --prompts 1 --trials 2 --repetitions 1 --warmup 1 --steps 2 --output local_checks/final_A
python benchmark.py --set B --device cpu --contexts 8 32 --prompts 1 --trials 2 --repetitions 1 --warmup 1 --steps 2 --output local_checks/final_B
python tests/validate_pretrained.py
python tests/trace_fullgraph.py
python tests/validate_sdpa.py
python tests/test_runner_protocol.py
python tests/validate_fresh_parameters.py

logs/ contains renewed evidence; local_checks/pretrained_extended.json has42 cases and all outputs. Source exported using eager attention; prepared/A and B include full baseline/optimized IR, generic bridge and audit hashes. A no rewrite; B60 matches. local_checks/offline_cuda.json and cuda_* folders contain successful sm_75 codegen/PTX. CUDA execution unavailable locally. Shared split-reduction/argmax flags are required for the B vocabulary reduction on this compiler. Torch compile GPU is not validated; its failures are explicitly logged and skipped as unavailable.

Next command on user's GPU: upload Colab_Set_A.ipynb (friend: B) and common ZIP, run notebook cells in order. Manual equivalent: python scripts/run_colab_set.py --set A (friend: B).
Next three tasks: (1) collect both final archives and inspect correctness/partial failures; (2) compare repeated within-account step latencies, SDPA/compiled Torch and token agreement; (3) if improvement survives, profile cache-commit traffic and test a representative real branching decoder/more architectures. Do not claim all ordinary GQA/greedy graphs benefit.

Files should survive restarts through saved Attention_Real_LLM_Tests.zip, library_file_id libfile_b7fc63608b6c8191b75bd0c924c141cd. Do not rely on scratch state. Notebook backups save every5minutes and at completion to each user's own Drive, then attempt browser download. A hard runtime kill can prevent final callbacks.
