# Update chains and observable snapshots

These extension measurements are a separate isolated CPU run. Compilation and output downloads are excluded; wrong stock snapshot code is not timed. See README.md for calibration and holdout methodology.

|Case|Best correct baseline µs|Old masked form µs|Coalescing alone µs|Combined masked µs|Speedup (95% bootstrap)|
|---|---:|---:|---:|---:|---:|
|T1|197.42|79.75|209.34|76.82|2.57× (2.34–2.89)|
|T4|207.85|214.99|221.28|88.81|2.34× (2.01–2.57)|
|T8|211.10|211.62|219.91|91.32|2.31× (1.98–2.63)|
|T16|233.63|234.87|245.63|107.59|2.17× (1.88–2.37)|
|Snapshot|109.01|incorrect without protection|—|42.27|2.58× (2.45–2.85)|

At T1, coalescing is a no-op and the existing forwarding rewrite already works. At T4–T16, the earlier rewrites do not match the update chains; coalescing makes the single-update representation available. Coalescing alone adds small temporary buffers and is not a speedup claim. Its benefit requires composition with forwarding.

|Case|Best-baseline invocation bytes|Combined invocation bytes|
|---|---:|---:|
|T1|3147776|3200|
|T4|3147776|7296|
|T8|2164736|11392|
|T16|3147776|19584|
|Snapshot|2098176|2176|

The snapshot case returns the pre-update K/V tensors. Stock IREE corrupts those declared outputs. Correct baselines include both early snapshot protection and the independently compiled Stream COW repair, with stronger fusion controls. The optimized form preserves the old caches while computing attention directly from the base/update rows, avoiding a full temporary cache. No speedup is computed against incorrect stock code.

The native failure also occurs with the separately installed IREE3.12.0 compiler. The first affected revision is not established. Seven compiler passes are compiled and exercised via the standalone MLIR bridge; the proposed in-tree patches have not both been built into a complete source compiler. GPU compilation is separate from GPU execution.

![Ablation](results/figures/extension-ablation.png)
