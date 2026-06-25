import torch
import sys
sys.path.insert(0, '.')
import ai8x
ai8x.set_device(device=85, simulate=True, round_avg=False)

from models.picosam3 import PicoSAM3
from datasets.coco_roi_seg_dataset import CocoRoiSegDataset

CKPT = './logs/2026.06.03-111153/quantized.pth.tar'
model = PicoSAM3(num_classes=1, bias=True)
ckpt = torch.load(CKPT, map_location='cpu')
result = model.load_state_dict(ckpt['state_dict'], strict=False)
print("Missing keys:", result.missing_keys)
print("Unexpected keys:", result.unexpected_keys)
model.eval()

for name, p in model.named_parameters():
    if 'weight' in name:
        print(f"{name}: min={p.min():.4f}, max={p.max():.4f}, mean={p.mean():.4f}")
        break

ds = CocoRoiSegDataset(root_dir='./datasets', split='val', image_size=80, output_size=20)
img, _ = ds[0]
print(f"\nInput shape: {img.shape}, range: [{img.min():.2f}, {img.max():.2f}]")

with torch.no_grad():
    out = model(img.unsqueeze(0))

print(f"Output shape: {out.shape}")
print(f"Output stats: min={out.min():.4f}, max={out.max():.4f}, mean={out.mean():.4f}")
print(f"Sigmoid output: min={torch.sigmoid(out).min():.4f}, max={torch.sigmoid(out).max():.4f}")