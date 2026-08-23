# YOLOEdge27 Stage 2 — Tiny Candidate Recovery v1

## Research question

Can tiny-object supervision be improved without changing the detector architecture by repairing the candidate pool used by Task-Aligned Assignment (TAL)?

## Motivation

The validated Stage-1 model already uses a P2/P3/P4 detection head and SPR-Down v1. Standard TAL ranks positive candidates whose anchor centers are inside each ground-truth box. For very small objects, that geometric constraint can leave only a few or even zero candidate centers, especially after resizing.

## TCR v1

Tiny Candidate Recovery (TCR) keeps standard TAL unchanged for normal objects. For a valid GT whose minimum side is <= 16 px on the 640-px training canvas:

1. Count the standard inside-GT anchor-center candidates across P2/P3/P4.
2. If the count is below 4, add the nearest unused anchor centers to reach 4 coarse candidates.
3. Run the original TAL classification/localization alignment metric and top-k ranking on that recovered pool.
4. Keep the original GT box as the regression target.

Only the coarse candidate pool changes. SPR-Down, Detect head, reg_max=16, CIoU+DFL, optimizer and augmentation remain unchanged.

## Stage-2 screening contract

- Base architecture: SPR-Down v1 at P4->P5.
- Detect: P2/P3/P4, strides 4/8/16.
- reg_max: 16.
- Loss: standard Ultralytics box/cls/DFL.
- Attention: none.
- Pretrained: false.
- Seed: 42.
- Local smoke: 5 epochs, batch 8 / nbs 16.
- Final paper evidence must be rerun on the locked Kaggle protocol.

## Diagnostic gate

Before training, measure candidate starvation on the converted VisDrone train/val sets using the exact 640 letterbox geometry. If fewer than 5% of tiny objects have <4 inside-GT candidates, the mechanism is not sufficiently motivated and the 5e smoke should be skipped/redesigned.

## Literature caution

This is an engineering hypothesis, not a novelty claim. Tiny-object label assignment is established research territory, including NWD-RKA, SimD, ATSS/OTA-style adaptive assignment, and dynamic coarse-to-fine methods. Any paper claim must distinguish TCR from these approaches and include assignment diagnostics/ablations rather than relying on a renamed top-k heuristic.
