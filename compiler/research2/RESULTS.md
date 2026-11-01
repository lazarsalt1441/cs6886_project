# Measured CPU results

> Experiment-stage record. See the repository root README for the final GPU evaluation and current reproduction entry point.

These are synthetic attention/block experiments on one AMD EPYC9V74 CPU, one thread. GPU runtime remains untested. Compilation is excluded. Read README.md for methodology and limitations.

|ID|Best baseline µs|Masked µs|Conditional µs|Calibration choice|Choice speedup (95% bootstrap)|
|---|---:|---:|---:|---|---:|
|A1|187.10|74.36|91.74|patch-forwarded|2.52× (2.39–2.72)|
|A2|256.06|271.21|158.51|forwarded|1.62× (1.52–1.68)|
|A3|3606.86|993.82|1713.41|patch-forwarded|3.63× (3.43–4.12)|
|A4|12981.46|131.10|238.55|patch-forwarded|99.02× (74.04–101.90)|
|A5|222.99|242.03|297.19|lifted-normalized|1.00× (0.93–1.08)|
|A6|2458.20|1733.45|1755.80|patch-forwarded|1.42× (1.37–1.51)|
|A7|95.18|66.95|75.53|patch-forwarded|1.42× (1.32–1.48)|
|A8|31.30|30.52|40.07|patch-forwarded|1.03× (0.91–1.17)|
|A9|61.28|63.82|87.53|patch-forwarded|0.96× (0.89–1.12)|
|B1|204.19|85.45|123.83|patch-forwarded|2.39× (2.06–2.68)|
|B2|1638.58|207.32|309.81|patch-forwarded|7.90× (7.24–8.40)|
|C1|2200.00|815.82|994.82|patch-forwarded|2.70× (2.46–2.90)|
|C2|230.97|228.35|241.93|lifted-normalized|0.99× (0.91–1.08)|
|W1|2607.34|723.42|835.03|factor-masked|3.60× (3.47–3.87)|

A1–A9 are the nine attention shapes in study_cache.py; B1/B2 are two synthetic full transformer blocks; C1 returns one fixed committed cache and C2 all cache versions. W1 returns the attention-derived winner cache and decision. The masked ratio2 ablation includes regressions; the revised default ratio4 gate skips A5. Calibration choice is not a universal static heuristic. Near1× no-op differences are timing noise. Bootstrap intervals describe invocation variability in this run.

|ID|Baseline allocation bytes|Masked allocation bytes|Baseline scheduled copies bytes|Masked scheduled copies bytes|
|---|---:|---:|---:|---:|
|A1|2164736|3200|4196352|0|
|A2|1347584|529536|4195328|0|
|A3|8658944|10496|67125248|0|
|A4|33622016|3200|67110912|0|
|A5|788480|6656|1050624|0|
|A6|12615680|33920|16785408|0|
|A7|875328|4160|1164672|0|
|A8|1024|1024|1024|1024|
|A9|197632|197632|264192|264192|
|B1|2103296|6784|4195328|0|
|B2|8399872|11264|33556480|0|
|C1|16846848|8393856|33558528|8389632|
|C2|4196352|4196352|4196352|4196352|
|W1|25168064|8391872|16779264|8388608|

Allocation bytes include workspace and outputs after inputs are uploaded. Scheduled copy lengths are not measured DRAM traffic. LLVM IR/disassembly and CUDA HAL schedules are retained separately. No GPU speedup or full pretrained-model throughput claim follows from these data. Raw incorrect gather code is excluded from all timing baselines.

CPU latency: `results/figures/cache-latency.png` (restored by `scripts/unpack_artifacts.py` from the repository root).

Allocation: `results/figures/cache-allocation.png` (restored by `scripts/unpack_artifacts.py` from the repository root).
