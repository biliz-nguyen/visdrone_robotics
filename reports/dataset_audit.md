# VisDrone2019-DET Dataset Audit

Source: official train / val / test-dev folders on the self-hosted runner.

## Split integrity

| Split | Images | Expected | Ann files | Missing ann | Orphan ann | Kept objects | Empty after filter |
|---|---:|---:|---:|---:|---:|---:|---:|
| train | 6471 | 6471 | 6471 | 0 | 0 | 343204 | 0 |
| val | 548 | 548 | 548 | 0 | 0 | 38759 | 0 |
| test-dev | 1610 | 1610 | 1610 | 0 | 0 | 75102 | 0 |

## Annotation health

| Split | Malformed | Invalid category | Non-positive box | Outside image | Duplicate rows | Ignored regions | Others |
|---|---:|---:|---:|---:|---:|---:|---:|
| train | 0 | 0 | 1 | 0 | 3 | 8813 | 1532 |
| val | 0 | 0 | 0 | 0 | 0 | 1378 | 32 |
| test-dev | 0 | 0 | 0 | 0 | 0 | 2180 | 265 |

## Small-object indicators

Definitions are pixel-space on original images; these are descriptive audit counts, not COCO size bins.

| Split | Area < 16x16 | Area < 32x32 | Min side < 16 px |
|---|---:|---:|---:|
| train | 89208 | 207524 | 126051 |
| val | 11950 | 26575 | 16935 |
| test-dev | 27341 | 50816 | 34879 |

## Train/val class distribution

| Class | Train | Val |
|---|---:|---:|
| pedestrian | 79337 | 8844 |
| people | 27059 | 5125 |
| bicycle | 10480 | 1287 |
| car | 144867 | 14064 |
| van | 24956 | 1975 |
| truck | 12875 | 750 |
| tricycle | 4812 | 1045 |
| awning-tricycle | 3246 | 532 |
| bus | 5926 | 251 |
| motor | 29647 | 4886 |
