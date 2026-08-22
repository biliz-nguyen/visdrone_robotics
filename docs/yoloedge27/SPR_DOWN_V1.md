# YOLOEdge27 Stage 1 — SPR-Down v1

## Working contribution name

**SPR-Down: Selective Phase Reassembly Downsampling**

This is the first clean contribution in the YOLOEdge27 research track. The legacy AConv, NWD and attention blocks are intentionally disabled for this stage so that the effect of SPR-Down can be measured independently.

## Research question

> Can phase-aware feature reassembly reduce tiny-object information loss and sampling sensitivity during edge-efficient stride-2 downsampling?

## Motivation

A stride-2 convolution reduces spatial resolution with a fixed sampling geometry. For tiny UAV targets, a one-pixel translation can materially change which samples contribute to the next feature level. SPR-Down treats stride-2 downsampling as four polyphase components and learns how to combine them before spatial/channel mixing.

## Tensor flow

For input

`X in R^(B x C x H x W)`

pad the right/bottom edge only when H or W is odd, then split:

- `X00 = X[:, :, 0::2, 0::2]`
- `X01 = X[:, :, 0::2, 1::2]`
- `X10 = X[:, :, 1::2, 0::2]`
- `X11 = X[:, :, 1::2, 1::2]`

Stacking gives

`P in R^(B x C x 4 x H/2 x W/2)`.

A sign-agnostic phase descriptor is

`D = mean(|P|, spatial)`

with shape `B x C x 4`.

The phase score is

`S = (D * gamma + beta) / tau`

where `gamma` and `beta` are learnable `1 x C x 4` tensors and `tau` is a fixed positive temperature.

Normalized weights are

`W = softmax(S, phase)`.

Reassembly is

`R[b,c,:,:] = sum_p W[b,c,p] * P[b,c,p,:,:]`.

The compact output transform is

`Y = PWConv1x1(SiLU(BN(DWConv3x3(R))))`.

The implementation applies BN + SiLU after both depthwise and pointwise convolutions.

## Why this differs from the old AConv

SPR-Down does not use fixed channel partitions such as Detail/Context/Preserve. It does not concatenate multiple convolution branches. Its central mechanism is input-conditioned reassembly of four sampling phases before lightweight convolutional mixing.

## Complexity

Ignoring BN, pooling, softmax and elementwise operations, a standard stride-2 3x3 convolution has approximately

`MAC_conv = H2 * W2 * 9 * C1 * C2`.

SPR-Down convolutional MACs are approximately

`MAC_spr = H2 * W2 * (9*C1 + C1*C2)`.

For the current P4->P5 n-scale location (`C1=128`, `C2=256`, `40x40 -> 20x20`):

- standard Conv3x3-s2: about **117.965 M MACs**;
- SPR-Down convolutional part: about **13.568 M MACs**;
- convolutional MAC ratio: about **0.115**.

Parameter count at this location, including BN affine parameters and the 8*C1 phase parameters:

- standard Conv3x3-s2 + BN: about **295,424 params**;
- SPR-Down v1: about **35,712 params**;
- parameter ratio: about **0.121**.

These are module-level figures, not whole-model GFLOPs. Actual latency must be measured because phase slicing, absolute value, global pooling, softmax and reassembly are not represented by the simple convolutional MAC estimate.

## Stage-1 experimental contract

The first fair comparison must keep everything except the selected downsampling block fixed:

- Detect levels: P2/P3/P4, strides 4/8/16;
- reg_max: 16;
- standard localization loss;
- no ECA/CA/RLCA;
- pretrained: false;
- seed: 42;
- same optimizer, image size, augmentation and epoch budget.

Initial pair:

1. `edge27_baseline`: standard Conv at P4->P5;
2. `edge27_spr_v1`: one SPR-Down at P4->P5.

Only after this pair is understood should SPR-Down be moved to earlier/multiple stages.

## Required measurements

Primary:

- mAP50-95;
- mAP50;
- pedestrian AP;
- people AP;
- Params;
- GFLOPs;
- training stability.

Edge-oriented:

- PyTorch latency;
- export success;
- Raspberry Pi latency/FPS later in the pipeline.

Mechanism-oriented follow-up:

- 1-pixel shift consistency on tiny objects;
- AP vs IoU threshold;
- AP by object size;
- ablation of scorer choice and temperature.

## Acceptance criteria for keeping SPR-Down

The block should be retained only if the accuracy/efficiency trade-off is meaningful. A useful initial target is:

- mAP50-95 no worse than roughly 0.2 AP point below the matched baseline, preferably equal or higher;
- lower Params/GFLOPs;
- no material degradation for pedestrian/people;
- measured latency does not contradict the theoretical compute advantage.

If the module fails these conditions, redesign the mechanism rather than stacking attention modules on top.
