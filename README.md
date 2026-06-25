# Edge Image Segmentation
This repository contains the source files used to produce my results in the Machine Learning on Microcontrollers course at ETH,
organised by deployment target:

```
repo_root/
├── ai8x/                  # MAX78000 (Analog Devices ai8x toolchain)
│   ├── training/          # Drop-in files for ai8x-training/
│   └── synthesis/         # Drop-in files for ai8x-synthesis/
└── picosam3/              # IMX500 (Raspberry Pi AI Camera) + plotting
```

The files here are **only the project-specific additions / modifications**.
To recreate the results you need the upstream toolchains:

- `ai8x-training` — https://github.com/analogdevicesinc/ai8x-training
- `ai8x-synthesis` — https://github.com/analogdevicesinc/ai8x-synthesis
- `picosam3` — https://github.com/pbonazzi/picosam3 (cloned by the Colab notebooks)

---

## 1. MAX78000 — Training & QAT (`ai8x/training/`)

These files are designed to be placed into a fresh clone of `ai8x-training`:

| File | Purpose | Where it goes |
|------|---------|----------------|
| [train.py](ai8x/training/train.py) | Modified upstream trainer — wires in the KD loss (see line 450) | `ai8x-training/train.py` (overwrite) |
| [kd_loss.py](ai8x/training/kd_loss.py) | Knowledge-distillation loss (soft MSE + BCE + Dice) | `ai8x-training/kd_loss.py` |
| [picosam3.py](ai8x/training/picosam3.py) | PyTorch model definition (folded U-Net with concat skips) | `ai8x-training/models/picosam3.py` |
| [coco_roi_seg_dataset.py](ai8x/training/coco_roi_seg_dataset.py) | COCO ROI segmentation dataset (loads cached teacher logits) | `ai8x-training/datasets/coco_roi_seg_dataset.py` |
| [picosam3_qat.yaml](ai8x/training/picosam3_qat.yaml) | QAT schedule (8-bit weights / activations / bias) | `ai8x-training/policies/picosam3_qat.yaml` |
| [metrics.py](ai8x/training/metrics.py) | Reports params / MAC / IoU on the val set | `ai8x-training/metrics.py` |
| [picosam3_vis.py](ai8x/training/picosam3_vis.py) | Renders a 4×3 grid of input / GT / student predictions → `predictions.png` | `ai8x-training/picosam3_vis.py` |
| [qual_compare.py](ai8x/training/qual_compare.py) | Renders student-vs-teacher qualitative comparison → `max78000_qualitative.png` | `ai8x-training/qual_compare.py` |
| [qual_diagnosis.py](ai8x/training/qual_diagnosis.py) | Sanity-check a quantized checkpoint (key match, weight stats) | `ai8x-training/qual_diagnosis.py` |
| [images/predictions.png](ai8x/training/images/predictions.png), [images/max78000_qualitative.png](ai8x/training/images/max78000_qualitative.png) | Reference output figures | — |

The teacher logits consumed by the dataset are produced by
[picosam3/colab_teacher_precompute.ipynb](picosam3/colab_teacher_precompute.ipynb)
and must be placed under `ai8x-training/datasets/teacher_sam3_logits/`,
alongside the COCO `train2017/`, `val2017/` and `annotations/` folders.

### Recreate the training results

```bash
# from ai8x-training/
source venv/bin/activate

# 1) QAT training with KD
#    (lr was varied across runs; 100 epochs for the reported result)
python train.py \
    --model picosam3 \
    --dataset coco_roi_seg \
    --epochs 100 \
    --optimizer Adam --lr 0.001 \
    --batch-size 64 \
    --qat-policy policies/picosam3_qat.yaml \
    --device MAX78000 \
    --use-bias

# 2) Quantize the QAT checkpoint
python quantize.py logs/<run>/qat_best.pth.tar logs/<run>/quantized.pth.tar --device MAX78000

# 3) Evaluation / figures (edit CKPT_PATH inside each script)
python metrics.py
python picosam3_vis.py
python qual_compare.py
python qual_diagnosis.py
```

---

## 2. MAX78000 — Synthesis (`ai8x/synthesis/`)

| File | Purpose |
|------|---------|
| [picosam3.yaml](ai8x/synthesis/picosam3.yaml) | Layer config for PicoSAM3 (concat-based skip connections via dual processor groups) |
| [make_sample.py](ai8x/synthesis/make_sample.py) | Generates the folded 12×40×40 sample input from a COCO val image |
| [sample_picosam_folded.npy](ai8x/synthesis/sample_picosam_folded.npy) | Pre-generated sample input used for the known-answer test |
| [ai8xize.py](ai8x/synthesis/ai8xize.py), [quantize.py](ai8x/synthesis/quantize.py) | Standard upstream entry points (unchanged, included for completeness) |

### Recreate the synthesis output

Place the project files inside the `ai8x-synthesis/` clone as follows:

- `picosam3.yaml` → `ai8x-synthesis/networks/picosam3.yaml`
- `sample_picosam_folded.npy` → `ai8x-synthesis/tests/sample_picosam_folded.npy`
  (auto-discovered via the `dataset: picosam_folded` field in the yaml)
- `make_sample.py` → `ai8x-synthesis/make_sample.py`

This assumes `ai8x-training` and `ai8x-synthesis` are sibling directories so
the `--checkpoint-file` relative path resolves.

```bash
# from ai8x-synthesis/
source venv/bin/activate

# (Optional) regenerate the sample input from a fresh COCO val image
python make_sample.py

# Synthesize C code for the MAX78000 SDK (actual command used)
python ai8xize.py \
    --test-dir demos \
    --prefix picosam3 \
    --checkpoint-file ../ai8x-training/logs/2026.06.02-172241/quantized.pth.tar \
    --config-file networks/picosam3.yaml \
    --device MAX78000 \
    --compact-data --mexpress --timer 0 \
    --display-checkpoint --verbose --overwrite
```

The `quantized.pth.tar` checkpoint is the output of step 2 (`quantize.py`) from
the training section above. Replace the `logs/2026.06.02-172241/` path with
the timestamped log directory from your own training run.

---

## 3. IMX500 deployment & plots (`picosam3/`)

These files target the Raspberry Pi AI Camera and assume the
`pbonazzi/picosam3` repo (cloned by the notebooks).

| File | Purpose |
|------|---------|
| [colab_teacher_precompute.ipynb](picosam3/colab_teacher_precompute.ipynb) | Precomputes SAM3 teacher logits and caches them to Drive |
| [colab_distillation.ipynb](picosam3/colab_distillation.ipynb) | Distillation training of PicoSAM3 against the cached teacher logits |
| [imx500_converter.py](picosam3/imx500_converter.py) | Sony MCT post-training quantization → ONNX (with optional pruning / mixed precision / bias correction flags) |
| [imx500_deployment.py](picosam3/imx500_deployment.py) | Live inference on the Raspberry Pi AI Camera with interactive ROI selection; records to MP4 |
| [plot_latency_vs_size.py](picosam3/plot_latency_vs_size.py) | Latency-vs-size scatter plot (PicoSAM3 vs other SAM variants) |
| [plot_map_vs_size.py](picosam3/plot_map_vs_size.py) | mAP-vs-size scatter (COCO and LVIS) |
| [plot_miou_vs_size.py](picosam3/plot_miou_vs_size.py) | mIoU-vs-size scatter |
| [wandb/](picosam3/wandb/) | Reference training-run figures (`bent_pizza.png`, `tennis_racket.png`, `training.png`) exported from W&B |

### Recreate the IMX500 results

```bash
# 1) (Colab) Precompute teacher logits — run colab_teacher_precompute.ipynb end-to-end
# 2) (Colab) Train the student — run colab_distillation.ipynb end-to-end
#    → produces checkpoints/PicoSAM3_SAM3_student_epoch10.pt

# 3) Convert to a quantized .onnx (run locally; needs model_compression_toolkit)
python imx500_converter.py

# 4) Package the .onnx as an .rpk using Sony's imx500-converter tool, then deploy:
python imx500_deployment.py --model /usr/share/imx500-models/picosam3_bm.rpk
#   Click-and-drag in the preview window to set the segmentation ROI.

# 5) Reproduce the result plots
python plot_latency_vs_size.py
python plot_map_vs_size.py
python plot_miou_vs_size.py
```

---

## 4. STM32 U5 — note

There are no project-specific files for the STM32 U5 because the deployment is
trivial: export `picosam3.onnx` using `imx500_converter.py` and import it directly into the **X-CUBE-AI**
pack in STM32CubeMX. 
---

## Environment notes

- **ai8x toolchain** — Python 3.11.x venv per the ai8x-training / ai8x-synthesis
  install instructions; use the venvs each upstream repo specifies.
- **IMX500 conversion** — `model_compression_toolkit`, `onnx`, `torch`,
  `pycocotools`; runs on CPU but a GPU speeds up MCT calibration.
- **IMX500 deployment** — Raspberry Pi 5 with the AI Camera, `picamera2`,
  `opencv-python`. The model must already be packaged as an `.rpk` (built from
  the `imxconv-pt` output zip produced at the end of `colab_distillation.ipynb`).
- **Plot scripts** — `matplotlib`, `numpy`, `adjustText` (latency plot only).
