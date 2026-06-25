"""
Knowledge Distillation Loss for PicoSAM3 (Probability-Based Soft Loss).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class KDLoss(nn.Module):
    def __init__(self, alpha=0.3, temperature=4.0, dice_weight=1.0, bce_weight=0.1):
        super().__init__()
        self.alpha = alpha
        self.T = temperature
        self.dice_weight = dice_weight
        self.bce_weight = bce_weight   # ← downweight BCE to balance scales

    def forward(self, student_logits, target):
        if student_logits.dim() == 4:
            student_logits = student_logits.squeeze(1)

        teacher_logits = target[:, 0]
        gt_mask = target[:, 1]

        T = self.T

        # Soft loss with temperature — smoothed probabilities
        student_soft = torch.sigmoid(student_logits / T)
        teacher_soft = torch.sigmoid(teacher_logits / T)
        soft_loss = F.mse_loss(student_soft, teacher_soft) * (T * T)

        # BCE on raw logits, then Dice on sigmoid output
        student_probs = torch.sigmoid(student_logits)
        bce_loss = F.binary_cross_entropy_with_logits(student_logits, gt_mask)
        intersection = (student_probs * gt_mask).sum(dim=(1, 2))
        union = student_probs.sum(dim=(1, 2)) + gt_mask.sum(dim=(1, 2))
        dice_loss = (1.0 - (2.0 * intersection + 1e-4) / (union + 1e-4)).mean()

        hard_loss = self.bce_weight * bce_loss + self.dice_weight * dice_loss
        return (1 - self.alpha) * soft_loss + self.alpha * hard_loss