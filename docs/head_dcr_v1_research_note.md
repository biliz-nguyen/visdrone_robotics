# DCR-Head v1 research note

Working name: **DCR-Head — Detail-Corrected Regression Head**.

This is an engineering hypothesis, not a novelty claim.

## Lessons retained from prior head designs

Recent lightweight detection heads repeatedly emphasize a few useful principles: decouple classification and localization, remove redundant heavy regression computation, preserve high-resolution detail for small objects, keep edge inference simple/exportable, and avoid applying expensive enhancement at every pyramid level when only a fine level may need it.

Nearest design families reviewed before implementation include YOLOX-style decoupled heads, TOOD task-aligned heads, shared/lightweight heads, level-selective regression towers, Shuffle/DWConv heads, and re-parameterized detail heads.

## Deliberately different mechanism

DCR-Head does **not** use per-level DFL bin counts, shared convolutions across pyramid levels, channel shuffle, depthwise regression towers, dynamic attention/deformable convolution, task-aligned feature decomposition, or re-parameterized multi-kernel convolutions.

Instead it starts from the efficient DFL-free direct-regression control and adds one **P2-only local-detail correction**:

1. All levels predict direct l/t/r/b distances with a compact one-convolution regression tower.
2. Only P2 forms a high-frequency detail residual `d = x - AvgPool3x3(x)`.
3. A cheap 1x1 projection maps `d` to four residual box corrections.
4. P2 final distance prediction is `base + gamma * tanh(delta)`, where `gamma` is a learnable scalar initialized conservatively.
5. P3/P4 remain pure direct regression.
6. Classification remains the stock YOLO11-style decoupled classification branch.

The intent is to retain the large compute saving observed in the global DFL-free H1 control while recovering tiny-object localization from P2 detail cues.

## Causal test

Reuse existing controls; do not retrain them:

- H0: S1 + stock reg16/DFL head.
- H1: S1 + stock direct reg1/DFL-free head.

Train only:

- D1: S1 + DCR-Head v1.

Initial screen: 5 epochs, same local protocol as the previous head study. Promote to 50e only if D1 gives a convincing accuracy/efficiency trade-off.
