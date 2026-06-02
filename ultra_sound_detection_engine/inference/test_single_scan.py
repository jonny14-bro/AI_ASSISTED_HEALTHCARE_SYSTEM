#test_single_scan.py
import os, sys
import torch
import cv2
import json
import numpy as np
import scipy.ndimage as ndi

PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..")
)
sys.path.insert(0, PROJECT_ROOT)

from ultra_sound_detection_engine.models.segmentation.attention_unet import AttentionUNet

# --------------------
# CONFIG
# --------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_PATH = os.path.join(
    PROJECT_ROOT,
    "healthcare_assistant",
    "ultra_sound_detection_engine",
    "checkpoints",
    "best_attention_unet.pth"
)

def ultrasound_scan(image_path, output_dir, patient_id):

    IMG_SIZE = 256
    SOFT_THRESHOLD = 0.15
    FINAL_THRESHOLD = 0.2

    os.makedirs(output_dir, exist_ok=True)

    # --------------------
    # LOAD MODEL
    # --------------------
    model = AttentionUNet(1, 1).to(DEVICE)
    model.load_state_dict(
        torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True)
    )
    model.eval()

    # --------------------
    # LOAD IMAGE
    # --------------------
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    assert img is not None, "Image not found"

    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img_norm = img.astype(np.float32) / 255.0
    tensor = torch.from_numpy(img_norm).unsqueeze(0).unsqueeze(0).to(DEVICE)

    # --------------------
    # INFERENCE
    # --------------------
    with torch.no_grad():
        prob_map = model(tensor)

    prob_np = torch.sigmoid(prob_map).squeeze().cpu().numpy()
    prob_np = np.clip(prob_np, 0.0, 1.0)


    # --------------------
    # POST-PROCESS (Largest Connected Component)
    # --------------------
    soft_binary = prob_np > SOFT_THRESHOLD
    labeled, num = ndi.label(soft_binary)

    if num > 0:
        sizes = ndi.sum(soft_binary, labeled, range(1, num + 1))
        lcc_mask = (labeled == (np.argmax(sizes) + 1))
    else:
        lcc_mask = soft_binary

    mask = ((prob_np > FINAL_THRESHOLD) & lcc_mask).astype(np.uint8)

    # --------------------
    # SAVE OUTPUT IMAGES
    # --------------------
    mask_path = os.path.join(output_dir, "mask.png")
    overlay_path = os.path.join(output_dir, "overlay.png")

    cv2.imwrite(mask_path, mask * 255)

    overlay = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    red_mask = np.zeros_like(overlay)
    red_mask[:, :, 2] = mask * 255

    alpha = 0.4
    overlay = cv2.addWeighted(overlay, 1.0, red_mask, alpha, 0)

    cv2.imwrite(overlay_path, overlay)

    confidence = float(prob_np.max() * 100)

    # --------------------
    # SAVE JSON (PATIENT-BOUND)
    # --------------------
    result = {
        "patient_id": patient_id,
        "modality": "ultrasound",
        "confidence": confidence,
        "mask_path": mask_path.replace("\\", "/"),
        "overlay_path": overlay_path.replace("\\", "/"),
        "image_size": IMG_SIZE,
        "model": "Attention U-Net v1"
    }

    json_path = os.path.join(output_dir, "result.json")

    with open(json_path, "w") as f:
        json.dump(result, f, indent=4)

    return result, json_path, output_dir
'''
# --------------------
# CONSOLE OUTPUT
# --------------------
    print(f"\nSession ID: {SESSION_ID}")
    print(f"Lesion confidence: {confidence:.2f}%")
    print(f"Results saved in: {OUTPUT_DIR}")

# --------------------
# OPTIONAL VISUALIZATION
# --------------------

plt.figure(figsize=(14, 4))

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
plt.imshow(mask, cmap="gray")
plt.axis("off")

plt.subplot(1, 4, 4)
plt.title("Overlay")
plt.imshow(img, cmap="gray")
plt.imshow(mask, cmap="Reds", alpha=0.45)
plt.axis("off")

plt.tight_layout()
plt.show()
'''