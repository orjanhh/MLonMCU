import torch
import matplotlib.pyplot as plt
import sys
sys.path.insert(0, '.')
import ai8x
ai8x.set_device(device=85, simulate=False, round_avg=False)
from models.picosam3 import PicoSAM3
from datasets.coco_roi_seg_dataset import CocoRoiSegDataset

model = PicoSAM3(num_classes=1, bias=True)
ckpt = torch.load('./logs/2026.06.01-103956/qat_best.pth.tar', map_location='cpu')
model.load_state_dict(ckpt['state_dict'], strict=False)
model.eval()

ds = CocoRoiSegDataset(root_dir='./datasets', split='val', image_size=80, output_size=20)

fig, axes = plt.subplots(4, 3, figsize=(9, 12))
for i in range(4):
    img, target = ds[i * 100]  # spread out samples
    teacher = target[0]
    gt = target[1]
    with torch.no_grad():
        pred = torch.sigmoid(model(img.unsqueeze(0)).squeeze()).numpy()
    
    img_vis = (img.permute(1,2,0) + 128).clip(0,255).numpy().astype('uint8')
    axes[i,0].imshow(img_vis); axes[i,0].set_title('Input')
    axes[i,1].imshow(gt, cmap='RdYlGn'); axes[i,1].set_title('GT')
    axes[i,2].imshow(pred, cmap='RdYlGn'); axes[i,2].set_title('Student')
    for ax in axes[i]: ax.axis('off')
plt.tight_layout()
plt.savefig('predictions.png', dpi=100)
print("Saved predictions.png")