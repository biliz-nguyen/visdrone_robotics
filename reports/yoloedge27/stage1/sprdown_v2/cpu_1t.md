# SPR-Down v2 latency gate — cpu

- PyTorch: `2.5.1+cu124`
- Device: `CPU (1 torch threads)`
- Warmup/iters/repeats: `30/150/7`
- Verdict: **PASS** — SPR-Down v2 is at least 20% faster than v1 at the current P4->P5 site.

| Site | Conv ms | SPR-v1 ms | SPR-v2 ms | v1/Conv | v2/Conv | v2/v1 |
|---|---:|---:|---:|---:|---:|---:|
| P4_to_P5 | 1.5896 | 0.4978 | 0.3931 | 0.313 | 0.247 | 0.790 |
| P3_to_P4 | 1.6405 | 1.2577 | 0.5366 | 0.767 | 0.327 | 0.427 |
| P2_to_P3 | 1.7982 | 2.3064 | 0.8579 | 1.283 | 0.477 | 0.372 |

> This is a local micro-benchmark for screening. It is not a Raspberry Pi 5 deployment result and must not be reported as final edge latency.
