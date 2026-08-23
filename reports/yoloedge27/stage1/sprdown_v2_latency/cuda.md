# SPR-Down v2 latency gate — cuda

- PyTorch: `2.5.1+cu124`
- Device: `NVIDIA GeForce RTX 4060 Ti`
- Warmup/iters/repeats: `100/500/9`
- Verdict: **PASS** — SPR-Down v2 is at least 20% faster than v1 at the current P4->P5 site.

| Site | Conv ms | SPR-v1 ms | SPR-v2 ms | v1/Conv | v2/Conv | v2/v1 |
|---|---:|---:|---:|---:|---:|---:|
| P4_to_P5 | 0.0385 | 0.0949 | 0.0598 | 2.467 | 1.554 | 0.630 |
| P3_to_P4 | 0.0335 | 0.0892 | 0.1363 | 2.663 | 4.072 | 1.529 |
| P2_to_P3 | 0.0372 | 0.0889 | 0.0977 | 2.394 | 2.629 | 1.098 |

> This is a local micro-benchmark for screening. It is not a Raspberry Pi 5 deployment result and must not be reported as final edge latency.
