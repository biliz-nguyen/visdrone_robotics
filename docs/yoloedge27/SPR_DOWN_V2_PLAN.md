# SPR-Down v2 — edge-latency prototype

## Why v2 exists

SPR-Down v1 completed 50-epoch local screening successfully and slightly improved the local validation metrics while reducing model parameters. However, its isolated CUDA micro-benchmark showed that the dynamic phase path (polyphase stack + global descriptor + softmax + weighted reassembly) can be slower than a standard stride-2 convolution even when theoretical convolutional MACs are much lower.

This makes runtime overhead the next problem to solve before expanding SPR-Down to more backbone stages.

## Working v2 idea

SPR-Down v2 keeps the phase-reassembly principle but removes input-conditioned GAP and softmax from the deployment path:

```
Input
  -> pad to even size
  -> pixel_unshuffle(2)
  -> grouped 1x1 Conv, groups=C, 4C -> C
  -> DWConv 3x3
  -> PWConv 1x1
  -> Output
```

The grouped 1x1 layer learns one four-phase reassembly kernel per original channel. Its weights are initialized to 0.25, so the initial reassembly is equivalent to 2x2 average pooling before being optimized by training.

## Research status

This is a working engineering hypothesis, not a novelty claim. Pixel-unshuffle / space-to-depth and polyphase sampling have substantial prior work, so a dedicated prior-art review is required before this design can be presented as a paper contribution.

## Gate before training

Do not train v2 until the prototype passes these checks:

1. Shape and gradient unit tests.
2. Initial phase reassembly equals 2x2 average pooling.
3. GPU and CPU micro-benchmark against standard Conv and SPR-Down v1.
4. Export smoke test after integration into Ultralytics.

The desired result is a material reduction in v1 runtime overhead. If v2 is not faster in measured runtime, it should not replace v1 merely because it has similar parameter/MAC counts.

## Next experimental sequence

After the v1 eval-only report is committed:

- benchmark v2 prototype on CUDA and CPU;
- if the latency gate passes, integrate `SPRDownV2` into the model builder and Ultralytics patch;
- run a short smoke training first;
- only then run the fair 50-epoch local comparison under the same Stage-1 protocol.
