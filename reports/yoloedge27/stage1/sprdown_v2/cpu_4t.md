# SPR-Down v2 latency gate — cpu

- PyTorch: `2.5.1+cu124`
- Device: `CPU (4 torch threads)`
- Warmup/iters/repeats: `30/150/7`
- Verdict: **BORDERLINE** — SPR-Down v2 is faster than v1, but the gain is below the 20% material-speedup gate.

| Site | Conv ms | SPR-v1 ms | SPR-v2 ms | v1/Conv | v2/Conv | v2/v1 |
|---|---:|---:|---:|---:|---:|---:|
| P4_to_P5 | 0.4444 | 0.2930 | 0.2368 | 0.659 | 0.533 | 0.808 |
| P3_to_P4 | 0.4743 | 0.7765 | 0.2748 | 1.637 | 0.579 | 0.354 |
| P2_to_P3 | 0.5248 | 0.4656 | 0.3778 | 0.887 | 0.720 | 0.811 |

> This is a local micro-benchmark for screening. It is not a Raspberry Pi 5 deployment result and must not be reported as final edge latency.
