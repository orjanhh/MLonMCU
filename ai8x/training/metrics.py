"""Params / MACs / IoU on the val set. Run from ai8x-training/."""
import torch
import numpy as np
import sys
sys.path.insert(0, '.')
import ai8x
ai8x.set_device(device=85, simulate=False, round_avg=False)

from models.picosam3 import PicoSAM3
from datasets.coco_roi_seg_dataset import CocoRoiSegDataset

CKPT_PATH = './logs/2026.06.03-115751/qat_best.pth.tar'

model = PicoSAM3(num_classes=1, bias=True)
ckpt = torch.load(CKPT_PATH, map_location='cpu')
model.load_state_dict(ckpt['state_dict'], strict=False)
model.eval()

total_params = sum(p.numel() for p in model.parameters())
print(f"Parameters:        {total_params:,}")
print(f"FP32 size:         {total_params * 4 / 1024:.1f} KB")
print(f"INT8 size:         {total_params / 1024:.1f} KB")

# Hand-counted per layer — thop misreports the ai8x fused conv/pool layers.
def conv_macs(in_ch, out_ch, k, h, w):
    return in_ch * out_ch * k * k * h * w

macs = 0
macs += conv_macs(12, 16, 3, 40, 40)   # enc1
macs += conv_macs(16, 32, 3, 20, 20)   # enc2 (after pool)
macs += conv_macs(32, 32, 3, 10, 10)   # enc3
macs += conv_macs(32, 48, 3,  5,  5)   # bottleneck
macs += conv_macs(48, 32, 3, 10, 10)   # dec3_up ConvTranspose
macs += conv_macs(64, 32, 3, 10, 10)   # dec3_conv (after concat)
macs += conv_macs(32, 32, 3, 20, 20)   # dec2_up ConvTranspose
macs += conv_macs(64, 32, 3, 20, 20)   # dec2_conv (after concat)
macs += conv_macs(32,  1, 1, 20, 20)   # output

print(f"MACs per inference: {macs:,} ({macs/1e6:.2f} M)")

val_ds = CocoRoiSegDataset(
    root_dir='./datasets', split='val',
    image_size=80, output_size=20,
)

def compute_iou(pred_mask, gt_mask, threshold=0.5):
    pred_bin = (pred_mask > threshold).float()
    intersection = (pred_bin * gt_mask).sum()
    union = pred_bin.sum() + gt_mask.sum() - intersection
    return (intersection / (union + 1e-6)).item()

ious = []
pixel_accs = []

n_samples = min(1000, len(val_ds))
with torch.no_grad():
    for i in range(n_samples):
        img, target = val_ds[i]
        gt = target[1]
        if gt.sum() < 1:
            continue
        pred = model(img.unsqueeze(0))
        pred_prob = torch.sigmoid(pred.squeeze())
        ious.append(compute_iou(pred_prob, gt))
        pred_bin = (pred_prob > 0.5).float()
        pixel_accs.append((pred_bin == gt).float().mean().item())

print(f"\nValidation results ({len(ious)} samples):")
print(f"  Mean IoU:        {np.mean(ious):.4f}")
print(f"  Median IoU:      {np.median(ious):.4f}")
print(f"  Pixel accuracy:  {np.mean(pixel_accs):.4f}")