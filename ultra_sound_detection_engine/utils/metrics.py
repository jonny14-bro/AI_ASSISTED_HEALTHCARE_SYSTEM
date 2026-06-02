#meterics.py
import torch

def dice_score(pred, target, threshold=0.7, smooth=1e-6):
    pred = (pred > threshold).float()
    target = target.float()

    intersection = (pred * target).sum()
    return (2 * intersection + smooth) / (
        pred.sum() + target.sum() + smooth
    )

