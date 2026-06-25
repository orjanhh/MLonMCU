"""Student vs teacher qualitative comparison. Run from ai8x-training/."""

import torch
import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.insert(0, '.')
import ai8x
ai8x.set_device(device=85, simulate=True, round_avg=False)  # simulate=True matches deployed INT8

from models.picosam3 import PicoSAM3
from datasets.coco_roi_seg_dataset import CocoRoiSegDataset

CKPT = './logs/2026.06.03-111153/qat_best.pth.tar'
model = PicoSAM3(num_classes=1, bias=True)
ckpt = torch.load(CKPT, map_location='cpu')
model.load_state_dict(ckpt['state_dict'], strict=False)
model.eval()

ds = CocoRoiSegDataset(root_dir='./datasets', split='val',
                       image_size=80, output_size=20)

sample_indices = [5, 27, 201]

fig, axes = plt.subplots(len(sample_indices), 4, figsize=(14, 3*len(sample_indices)))
if len(sample_indices) == 1:
    axes = axes.reshape(1, -1)

with torch.no_grad():
    for row, idx in enumerate(sample_indices):
        img_tensor, target = ds[idx]
        # target[0] = teacher logits, target[1] = GT mask
        teacher_logits = target[0]
        gt_mask = target[1]

        student_logits = model(img_tensor.unsqueeze(0)).squeeze()
        student_prob = torch.sigmoid(student_logits)

        # undo the ai8x −128 shift, transpose CHW → HWC
        rgb_display = (img_tensor.numpy().transpose(1, 2, 0) + 128) / 255.0
        rgb_display = np.clip(rgb_display, 0, 1)

        teacher_mask = torch.sigmoid(teacher_logits).numpy()

        axes[row, 0].imshow(rgb_display)
        axes[row, 0].set_title(f"Input ROI (sample {idx})")
        axes[row, 0].axis('off')

        axes[row, 1].imshow(student_prob.numpy(), cmap='gray', vmin=0, vmax=1)
        axes[row, 1].set_title("Student (INT8, deployed)")
        axes[row, 1].axis('off')

        axes[row, 2].imshow(teacher_mask, cmap='gray', vmin=0, vmax=1)
        axes[row, 2].set_title("Teacher (SAM3)")
        axes[row, 2].axis('off')

        axes[row, 3].imshow(gt_mask.numpy(), cmap='gray', vmin=0, vmax=1)
        axes[row, 3].set_title("Ground truth")
        axes[row, 3].axis('off')

plt.tight_layout()
plt.savefig('max78000_qualitative.png', dpi=120, bbox_inches='tight')
print("Saved max78000_qualitative.png")