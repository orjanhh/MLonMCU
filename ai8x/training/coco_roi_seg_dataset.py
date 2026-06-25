"""
COCO ROI Segmentation Dataset for PicoSAM3 on MAX78000
Updated for 80x80 input / 20x20 output (MAX78000 memory constraint).
"""

import os
import glob
import torch
import torch.nn.functional as F
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
from pycocotools.coco import COCO
from pycocotools import mask as coco_mask

import ai8x


class CocoRoiSegDataset(Dataset):
    def __init__(
        self,
        root_dir,
        split="train",
        image_size=80,
        output_size=20,
        transform=None,
        per_image_cache=False,
        augment=False,
        max_samples=None,
    ):
        self.root_dir = root_dir
        self.image_size = image_size
        self.output_size = output_size
        self.transform = transform
        self.per_image_cache = per_image_cache
        self.augment = augment and (split == "train")
        self.split = split

        self.cache_dir = os.path.join(root_dir, "teacher_sam3_logits")
        self.train_img_dir = os.path.join(root_dir, "train2017")
        self.val_img_dir = os.path.join(root_dir, "val2017")

        train_ann = os.path.join(root_dir, "annotations", "instances_train2017.json")
        val_ann = os.path.join(root_dir, "annotations", "instances_val2017.json")

        print(f"Loading COCO annotations (train + val merged)...")
        self.coco = COCO(val_ann)
        if os.path.exists(train_ann):
            coco_train = COCO(train_ann)
            self.coco.imgs.update(coco_train.imgs)
            self.coco.anns.update(coco_train.anns)
            print(f"  Merged image index: {len(self.coco.imgs)} images")
            print(f"  Merged annotation index: {len(self.coco.anns)} annotations")

        self.samples = self._build_sample_index()

        if max_samples is not None and len(self.samples) > max_samples:
            import random
            random.seed(42)
            random.shuffle(self.samples)
            self.samples = self.samples[:max_samples]

        print(f"Dataset ready: {len(self.samples)} samples (split={split})")

    def _build_sample_index(self):
        samples = []
        if self.per_image_cache:
            cache_files = sorted(glob.glob(os.path.join(self.cache_dir, "img_*.pt")))
            print(f"Found {len(cache_files)} cached image files (lazy index)...")
            for cache_path in cache_files:
                image_id = int(os.path.basename(cache_path).replace("img_", "").replace(".pt", ""))
                samples.append({"image_id": image_id, "ann_id": None, "cache_path": cache_path})
        else:
            cache_files = sorted(glob.glob(os.path.join(self.cache_dir, "ann_*.pt")))
            print(f"Found {len(cache_files)} cached annotation files (lazy index)...")
            for cache_path in cache_files:
                ann_id = int(os.path.basename(cache_path).replace("ann_", "").replace(".pt", ""))
                samples.append({"image_id": None, "ann_id": ann_id, "cache_path": cache_path})
        return samples

    def _load_cache(self, sample):
        try:
            return torch.load(sample["cache_path"], map_location="cpu", weights_only=True)
        except TypeError:
            return torch.load(sample["cache_path"], map_location="cpu")
        except Exception:
            return None

    def _find_image_path(self, file_name):
        for img_dir in (self.train_img_dir, self.val_img_dir):
            candidate = os.path.join(img_dir, file_name)
            if os.path.exists(candidate):
                return candidate
        return None

    def _rasterize_gt(self, ann_id, image_id):
        if ann_id is None or ann_id not in self.coco.anns:
            return None
        ann = self.coco.anns[ann_id]
        img_info = self.coco.imgs.get(image_id)
        if img_info is None:
            return None
        h, w = img_info["height"], img_info["width"]
        try:
            if isinstance(ann["segmentation"], list):
                rles = coco_mask.frPyObjects(ann["segmentation"], h, w)
                rle = coco_mask.merge(rles)
            elif isinstance(ann["segmentation"], dict):
                rle = ann["segmentation"]
            else:
                return None
            return coco_mask.decode(rle)
        except Exception:
            return None

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        cache_data = self._load_cache(sample)
        if cache_data is None:
            return self._dummy_sample()

        if self.per_image_cache:
            ann_id = next(iter(cache_data.keys()))
            ann_data = cache_data[ann_id]
            image_id = sample["image_id"]
        else:
            ann_data = cache_data
            image_id = ann_data["image_id"]
            ann_id = sample["ann_id"]

        if image_id not in self.coco.imgs:
            return self._dummy_sample()

        img_info = self.coco.imgs[image_id]
        file_name = img_info.get("file_name", f"{image_id:012d}.jpg")
        img_path = self._find_image_path(file_name)
        if img_path is None:
            return self._dummy_sample()

        try:
            image = Image.open(img_path).convert("RGB")
            image.load()
            img_np = np.array(image)
        except Exception:
            return self._dummy_sample()

        rx1, ry1, rx2, ry2 = ann_data["roi"]
        h, w = img_np.shape[:2]
        rx1, ry1 = max(0, rx1), max(0, ry1)
        rx2, ry2 = min(w, rx2), min(h, ry2)
        if rx2 <= rx1 or ry2 <= ry1:
            return self._dummy_sample()

        roi_crop = img_np[ry1:ry2, rx1:rx2]
        if roi_crop.size == 0:
            return self._dummy_sample()

        try:
            roi_pil = Image.fromarray(roi_crop)
            roi_pil = roi_pil.resize((self.image_size, self.image_size), Image.BILINEAR)
            roi_np = np.array(roi_pil).astype(np.float32)
        except Exception:
            return self._dummy_sample()

        roi_tensor = torch.from_numpy(roi_np).permute(2, 0, 1)
        roi_tensor = roi_tensor.sub(128.0).clamp(-128.0, 127.0)

        teacher_logits = ann_data["logits"].float()
        if self.output_size != teacher_logits.shape[-1]:
            teacher_logits = F.interpolate(
                teacher_logits.unsqueeze(0),
                size=(self.output_size, self.output_size),
                mode="bilinear", align_corners=False,
            ).squeeze(0)
        teacher_target = teacher_logits.squeeze(0)

        gt_full = self._rasterize_gt(ann_id, image_id)
        if gt_full is None:
            gt_target = (torch.sigmoid(teacher_target) > 0.5).float()
        else:
            gt_roi = gt_full[ry1:ry2, rx1:rx2]
            if gt_roi.size == 0:
                gt_target = (torch.sigmoid(teacher_target) > 0.5).float()
            else:
                gt_pil = Image.fromarray((gt_roi * 255).astype(np.uint8))
                gt_pil = gt_pil.resize((self.output_size, self.output_size), Image.NEAREST)
                gt_target = torch.from_numpy(np.array(gt_pil)).float() / 255.0

        if self.augment and torch.rand(1).item() > 0.5:
            roi_tensor = torch.flip(roi_tensor, dims=[2])
            teacher_target = torch.flip(teacher_target, dims=[1])
            gt_target = torch.flip(gt_target, dims=[1])

        if self.transform is not None:
            roi_tensor = self.transform(roi_tensor)

        stacked_target = torch.stack([teacher_target, gt_target], dim=0)
        return roi_tensor, stacked_target

    def _dummy_sample(self):
        roi = torch.zeros(3, self.image_size, self.image_size, dtype=torch.float32)
        tgt = torch.zeros(2, self.output_size, self.output_size, dtype=torch.float32)
        return roi, tgt


def coco_roi_seg_get_datasets(data, load_train=True, load_test=True, **kwargs):
    if isinstance(data, tuple):
        data = data[0]

    train_dataset = None
    test_dataset = None

    if load_train:
        train_dataset = CocoRoiSegDataset(
            root_dir=data, split="train", image_size=80, output_size=20,
            per_image_cache=False, augment=True,
        )
    if load_test:
        test_dataset = CocoRoiSegDataset(
            root_dir=data, split="val", image_size=80, output_size=20,
            per_image_cache=False, augment=False,
        )
    return train_dataset, test_dataset


datasets = [
    {
        "name": "coco_roi_seg",
        "input": (3, 80, 80),
        "output": (1, 20, 20),
        "loader": coco_roi_seg_get_datasets,
        "regression": True,
    }
]