import os
import sys
import torch
from torch.utils.data import DataLoader
from torch.amp import GradScaler
from torch import autocast

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from models.segmentation.attention_unet import AttentionUNet
from utils.losses import DiceFocalLoss
from utils.metrics import dice_score
from data.ultrasound_dataset import UltrasoundSegmentationDataset

# --------------------
# CONFIG
# --------------------
IMAGES_DIR = "data/processed/images"
MASKS_DIR = "data/processed/masks"

BATCH_SIZE = 4
EPOCHS = 50
LR = 5e-5 # 🔧 FIX 1: LR = 1e-4 stronger LR for Attention U-Net

os.makedirs("checkpoints", exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
USE_AMP = DEVICE == "cuda"

# --------------------
# DATA
# --------------------
train_dataset = UltrasoundSegmentationDataset(
    images_dir=IMAGES_DIR,
    masks_dir=MASKS_DIR,
    mode="train"
)

val_dataset = UltrasoundSegmentationDataset(
    images_dir=IMAGES_DIR,
    masks_dir=MASKS_DIR,
    mode="val"
)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0,
    pin_memory=USE_AMP
)

val_loader = DataLoader(
    val_dataset,
    batch_size=1,
    shuffle=False,
    num_workers=0,
    pin_memory=USE_AMP
)

# --------------------
# MODEL
# --------------------
model = AttentionUNet(in_channels=1, out_channels=1).to(DEVICE)
criterion = DiceFocalLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
scaler = GradScaler(enabled=USE_AMP)

best_val_dice = 0.0

# --------------------
# TRAIN + VALIDATION
# --------------------
for epoch in range(1, EPOCHS + 1):

    # ---------- TRAIN ----------
    model.train()
    train_loss = 0.0

    for images, masks in train_loader:
        images = images.to(DEVICE, non_blocking=True)
        masks = masks.to(DEVICE, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with autocast(device_type=DEVICE, enabled=USE_AMP):
            logits = model(images)
            loss = criterion(logits, masks)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        train_loss += loss.item()

    train_loss /= max(len(train_loader), 1)

    # ---------- VALIDATION ----------
    model.eval()
    val_loss = 0.0
    val_dice = 0.0
    mean_prob = 0.0

    with torch.no_grad():
        for images, masks in val_loader:
            images = images.to(DEVICE, non_blocking=True)
            masks = masks.to(DEVICE, non_blocking=True)

            logits = model(images)
            loss = criterion(logits, masks)

            probs = torch.sigmoid(logits)

            # 🔧 FIX 2: stable Dice threshold
            val_dice += dice_score(probs, masks, threshold=0.5).item()
            val_loss += loss.item()
            mean_prob += probs.mean().item()


    val_loss /= max(len(val_loader), 1)
    val_dice /= max(len(val_loader), 1)
    mean_prob /= max(len(val_loader), 1)

    print(
        f"[Epoch {epoch:02d}/{EPOCHS}] "
        f"Train Loss: {train_loss:.4f} | "
        f"Val Loss: {val_loss:.4f} | "
        f"Val Dice: {val_dice:.4f} | "
        f"Mean Prob: {mean_prob:.3f}"
    )
    print("Pred mask sum:", (probs > 0.7).float().sum().item())

    

    # ---------- SAVE BEST ----------
    if val_dice > best_val_dice:
        best_val_dice = val_dice
        torch.save(
            model.state_dict(),
            "checkpoints/best_attention_unet.pth"
        )
        print("✅ Best model saved")
    
print(f"\n🏁 Training complete. Best Val Dice: {best_val_dice:.4f}")
        
