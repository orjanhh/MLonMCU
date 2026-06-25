"""
Generate sample input data for ai8x synthesis.
For PicoSAM3: 3x80x80 input, software-folded to 12x40x40.
Saves as int64 to tests/sample_picosam_folded.npy
"""

import numpy as np
from PIL import Image
import os

os.makedirs('tests', exist_ok=True)

# Load a representative validation image
img_path = '../ai8x-training/datasets/val2017/000000000139.jpg'
img = Image.open(img_path).convert('RGB')
img = img.resize((80, 80))

# Convert to model input format: CHW, shifted to int8 range
arr = np.array(img).transpose(2, 0, 1).astype(np.int64) - 128

# Fold (3, 80, 80) -> (12, 40, 40)
folded = arr.reshape(3, 40, 2, 40, 2)
folded = folded.transpose(0, 2, 4, 1, 3)
folded = folded.reshape(12, 40, 40)

np.save('tests/sample_picosam_folded.npy', folded)
print(f'Saved folded sample: shape {folded.shape}, dtype {folded.dtype}')
print(f'Range: [{folded.min()}, {folded.max()}]')