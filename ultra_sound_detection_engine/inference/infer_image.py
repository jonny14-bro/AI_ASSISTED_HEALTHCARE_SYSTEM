import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
import scipy.ndimage as ndi

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from models.segmentation.attention_unet import AttentionUNet

# --------------------
# CONFIG
# --------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_PATH = "checkpoints/best_attention_unet.pth"
IMAGE_PATH = "data/processed/images/train/malignant (1).png"
IMG_SIZE = 256
SOFT_THRESHOLD = 0.15   # for component extraction
FINAL_THRESHOLD = 0.2  # for final mask

# --------------------
# LOAD MODEL
# --------------------
model = AttentionUNet(in_channels=1, out_channels=1).to(DEVICE)
model.load_state_dict(
    torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True)
)
model.eval()

# --------------------
# LOAD IMAGE
# --------------------
img = cv2.imread(IMAGE_PATH, cv2.IMREAD_GRAYSCALE)
img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
img_norm = img.astype(np.float32) / 255.0

tensor = torch.from_numpy(img_norm).unsqueeze(0).unsqueeze(0).to(DEVICE)

# --------------------
# INFERENCE
# --------------------
with torch.no_grad():
    probs = model(tensor)

#prob_np = probs.squeeze().cpu().numpy()

# --------------------
# STEP 3 (CORRECT): LCC on SOFT probability map
# --------------------
prob_np = probs.squeeze().cpu().numpy()
prob_norm = (prob_np - prob_np.min()) / (prob_np.max() - prob_np.min() + 1e-8)

soft_binary = prob_norm > SOFT_THRESHOLD
labeled, num = ndi.label(soft_binary)

if num > 0:
    sizes = ndi.sum(soft_binary, labeled, range(1, num + 1))
    largest_label = np.argmax(sizes) + 1
    lcc_mask = (labeled == largest_label)
else:
    lcc_mask = soft_binary

# Final threshold AFTER component selection
mask_np = ((prob_norm > FINAL_THRESHOLD) & lcc_mask).astype(np.uint8)

# --------------------
# VISUALIZATION (NO OVERLAP)
# --------------------
plt.figure(figsize=(16, 4))

plt.subplot(1, 4, 1)
plt.title("Ultrasound")
plt.imshow(img, cmap="gray")
plt.axis("off")

plt.subplot(1, 4, 2)
plt.title("Probability Map")
plt.imshow(prob_np, cmap="jet")
plt.axis("off")

plt.subplot(1, 4, 3)
plt.title("Predicted Mask")
plt.imshow(mask_np, cmap="gray")
plt.axis("off")

plt.subplot(1, 4, 4)
plt.title("Overlay")
plt.imshow(img, cmap="gray")
plt.imshow(mask_np, cmap="jet", alpha=0.5)
plt.axis("off")

plt.tight_layout()
plt.show()
