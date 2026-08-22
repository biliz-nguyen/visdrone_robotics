# VisDrone Local Modular Pipeline (Ubuntu)

Project này được tách từ notebook `journal-q2.ipynb` thành các file riêng để chạy local trên Ubuntu.

## Mục tiêu

- Dataset path nằm **một chỗ duy nhất**: `config/local.yaml`.
- Experiment/module nằm **một chỗ duy nhất**: `config/experiment.yaml`.
- Có preset để đổi nhanh:
  - `baseline`
  - `reg8`
  - `reg4`
  - `reg4_eca`
  - `reg4_ca`
  - `reg4_rlca`
- Tự kiểm tra `reg_max`, Detect.no, stride, AConv, attention và loss trước train.
- Tên run sinh từ **config thật**, tránh trường hợp folder ghi `reg4` nhưng model thật là `reg8`.
- `tasks.py` và `loss.py` của Ultralytics được reset sạch trước mỗi lần patch.
- Train scratch: **không COCO pretrained**, không `model.load("yolo11n.pt")`.
- Có script riêng cho:
  - setup
  - sanity check
  - train
  - validate/test
  - test 1 ảnh
  - export NCNN / ONNX

---

## 1. Cấu trúc thư mục

```text
visdrone_local_project/
├── README.md
├── requirements.txt
├── setup_local.sh
├── run.sh
├── config/
│   ├── experiment.yaml
│   └── local.example.yaml
├── src/
│   ├── __init__.py
│   ├── config.py
│   ├── custom_blocks.py
│   ├── data.py
│   ├── model_builder.py
│   ├── ultralytics_patch.py
│   └── runtime.py
├── scripts/
│   ├── prepare.py
│   ├── sanity.py
│   ├── train.py
│   ├── validate.py
│   ├── infer.py
│   └── export.py
├── third_party/              # tự tạo khi setup
├── generated/                # data/model yaml tự sinh
├── runs/                     # kết quả train
├── state/                    # lưu best.pt của từng experiment
└── outputs/                  # export / prediction phụ
```

---

## 2. Chỗ duy nhất cần Gemini sửa DATA

Copy:

```bash
cp config/local.example.yaml config/local.yaml
```

Sau đó chỉ sửa:

```yaml
dataset_root: "/DUONG/DAN/THAT/TOI/VISDRONE"
```

Dataset code hiện kỳ vọng cấu trúc:

```text
DATASET_ROOT/
├── train/
│   ├── images/
│   └── labels/
├── valid/
│   ├── images/
│   └── labels/
└── test/
    ├── images/
    └── labels/
```

Nếu dataset local của bạn khác tên `valid`, Gemini chỉ cần sửa `train_images`, `val_images`, `test_images` trong `config/local.yaml`, không cần sửa code Python.

---

## 3. Setup Ubuntu

Khuyến nghị chạy trong Python environment đang có PyTorch/CUDA đúng với máy.

```bash
cd visdrone_local_project
bash setup_local.sh
```

Script sẽ clone Ultralytics đúng tag `v8.4.56` vào:

```text
third_party/ultralytics
```

và install editable.

> `detached HEAD` khi checkout tag là bình thường.

---

## 4. Chọn experiment

Mở:

```text
config/experiment.yaml
```

và sửa duy nhất:

```yaml
preset: reg4_eca
```

Các preset:

```text
baseline    = Conv, reg16, standard CIoU+DFL, no attention
reg8        = AConv, reg8, CIoU75+NWD25+DFL, no attention
reg4        = AConv, reg4, CIoU75+NWD25+DFL, no attention
reg4_eca    = reg4 + ECA after P2 C3k2
reg4_ca     = reg4 + Coordinate Attention after P2 C3k2
reg4_rlca   = reg4 + Residual Lite CA after P2 C3k2
```

Screening:

```yaml
epochs: 50
```

Final:

```yaml
epochs: 100
```

---

## 5. Chạy pipeline

### Prepare

```bash
python scripts/prepare.py
```

### Sanity check trước train

```bash
python scripts/sanity.py
```

Phải thấy đại loại:

```text
REAL reg_max: 4
REAL Detect.no: 26
Strides: [4, 8, 16]
AConv count: 1
Attention: ECA
Loss: hybrid_nwd
SANITY CHECK PASSED
```

### Train

```bash
python scripts/train.py
```

### Validate test split

```bash
python scripts/validate.py
```

Script tự lấy `best.pt` của experiment hiện tại từ `state/`.

Có thể cấp weight riêng:

```bash
python scripts/validate.py --weights /path/to/best.pt
```

### Test 1 ảnh

Tự lấy ảnh đầu tiên trong test:

```bash
python scripts/infer.py
```

Hoặc:

```bash
python scripts/infer.py --source /path/to/image.jpg
```

Ảnh prediction được lưu vào `outputs/predict/`.

### Export

NCNN:

```bash
python scripts/export.py --format ncnn
```

ONNX:

```bash
python scripts/export.py --format onnx
```

Cả hai:

```bash
python scripts/export.py --format ncnn onnx
```

---

## 6. Protocol hiện tại

```text
imgsz       = 640
batch       = 16
nbs         = 16
seed        = 42
pretrained  = False
optimizer   = AdamW
lr0         = 0.0015
lrf         = 0.05
momentum    = 0.937
weight_decay= 0.05
warmup      = 3
cos_lr      = True
max_det     = 500

hsv_h       = 0.015
hsv_s       = 0.5
hsv_v       = 0.3
translate   = 0.10
scale       = 0.35
fliplr      = 0.5
mosaic      = 1.0
close_mosaic= 10

degrees/shear/perspective/flipud = 0
mixup/copy_paste/cutmix = 0

box/cls/dfl = 7.5 / 0.5 / 1.5
```

`erasing=0.4` đã được bỏ khỏi local pipeline vì đây là detection pipeline.

---

## 7. Lưu ý paper

`python scripts/validate.py` dùng evaluator của Ultralytics để screening/so nội bộ.

Khi chốt số trong paper, vẫn chạy evaluator VisDrone chính thức theo cùng ignore-region/class-mapping/max_det protocol của bài.


## 8. Lệnh nhanh

```bash
./run.sh sanity
./run.sh train
./run.sh validate
./run.sh infer
./run.sh export-ncnn
```
