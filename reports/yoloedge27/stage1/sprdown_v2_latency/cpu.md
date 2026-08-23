# SPR-Down v2 latency gate — cpu

- PyTorch: `2.5.1+cu124`
- Device: `CPU (4 torch threads)`
- Warmup/iters/repeats: `30/150/7`
- Verdict: **BORDERLINE** — SPR-Down v2 is faster than v1, but the gain is below the 20% material-speedup gate.

| Site | Conv ms | SPR-v1 ms | SPR-v2 ms | v1/Conv | v2/Conv | v2/v1 |
|---|---:|---:|---:|---:|---:|---:|
| P4_to_P5 | 0.4764 | 0.2915 | 0.2884 | 0.612 | 0.605 | 0.990 |
| P3_to_P4 | 0.4675 | 0.5785 | 0.3629 | 1.237 | 0.776 | 0.627 |
| P2_to_P3 | 0.5210 | 0.4622 | 0.6157 | 0.887 | 1.182 | 1.332 |

> This is a local micro-benchmark for screening. It is not a Raspberry Pi 5 deployment result and must not be reported as final edge latency.
