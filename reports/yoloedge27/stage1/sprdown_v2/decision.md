# SPR-Down v2 benchmark decision

**Overall:** `BORDERLINE`

Keep v2 as candidate; inspect full-model/export latency before training 50e.

| Device | Gate | v1/Conv | v2/Conv | v2/v1 |
|---|---|---:|---:|---:|
| cuda | PASS | 2.272 | 1.252 | 0.551 |
| cpu_1t | PASS | 0.313 | 0.247 | 0.790 |
| cpu_4t | BORDERLINE | 0.659 | 0.533 | 0.808 |

> Local micro-benchmark only. Final edge claims require the exact final checkpoint on Raspberry Pi 5.
