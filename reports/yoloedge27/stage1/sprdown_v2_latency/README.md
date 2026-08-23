# SPR-Down v2 combined latency gate

CUDA verdict: **PASS** — SPR-Down v2 is at least 20% faster than v1 at the current P4->P5 site.
CPU verdict: **BORDERLINE** — SPR-Down v2 is faster than v1, but the gain is below the 20% material-speedup gate.

Primary site: P4->P5 (128->256, 40x40).

> CUDA is the primary local architecture-screening gate. CPU here is an x86 local screening result only; final edge evidence must be measured on the exact Raspberry Pi 5 export.
