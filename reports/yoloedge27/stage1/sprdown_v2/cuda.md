# SPR-Down v2 latency gate — cuda

- PyTorch: `2.5.1+cu124`
- Device: `NVIDIA GeForce RTX 4060 Ti`
- Warmup/iters/repeats: `50/300/7`
- Verdict: **PASS** — SPR-Down v2 is at least 20% faster than v1 at the current P4->P5 site.

| Site | Conv ms | SPR-v1 ms | SPR-v2 ms | v1/Conv | v2/Conv | v2/v1 |
|---|---:|---:|---:|---:|---:|---:|
| P4_to_P5 | 0.0392 | 0.0890 | 0.0490 | 2.272 | 1.252 | 0.551 |
| P3_to_P4 | 0.0320 | 0.0902 | 0.0493 | 2.820 | 1.542 | 0.547 |
| P2_to_P3 | 0.0362 | 0.0866 | 0.0479 | 2.390 | 1.324 | 0.554 |

> This is a local micro-benchmark for screening. It is not a Raspberry Pi 5 deployment result and must not be reported as final edge latency.
