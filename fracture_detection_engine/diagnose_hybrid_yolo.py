import torch
import torch.nn as nn
from torchvision import models, transforms
from ultralytics import YOLO
from PIL import Image

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# =====================================================
# 🔹 PATHS
# =====================================================

YOLO_MODEL_PATH = r"D:\healthcare_assistant\fracture_detection_engine\yolo_fracture.pt"

BODY_MODEL_PATH = r"D:\healthcare_assistant\fracture_detection_engine\bodyparts\body_part_densenet121.pth"

FRACTURE_MODELS = {
    "forearm": r"D:\healthcare_assistant\fracture_detection_engine\forearm\forearm_fracture_densenet121.pth",
    "wrist":   r"D:\healthcare_assistant\fracture_detection_engine\wrist\wrist_fracture_densenet121.pth",
    "leg":     r"D:\healthcare_assistant\fracture_detection_engine\leg\leg_fracture_densenet121.pth",
    "thigh":   r"D:\healthcare_assistant\fracture_detection_engine\thigh\thigh_fracture_densenet121.pth",
}

BODY_CLASSES = ["forearm", "leg", "thigh", "wrist"]
FRACTURE_CLASSES = ["fracture", "normal"]

# =====================================================
# 🔹 TRANSFORM
# =====================================================

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

# =====================================================
# 🔹 LOAD MODELS
# =====================================================

def load_model(path, num_classes):
    model = models.densenet121(weights=None)
    model.classifier = nn.Linear(1024, num_classes)
    model.load_state_dict(torch.load(path, map_location=DEVICE, weights_only=True))
    model = model.to(DEVICE)
    model.eval()
    return model

yolo_model = YOLO(YOLO_MODEL_PATH)

body_model = load_model(BODY_MODEL_PATH, len(BODY_CLASSES))

fracture_models = {
    part: load_model(path, 2)
    for part, path in FRACTURE_MODELS.items()
}

# =====================================================
# 🔹 DIAGNOSIS
# =====================================================

def diagnose(image_path):

    image = Image.open(image_path).convert("RGB")

    # ---- STEP 1: YOLO DETECTION ----
    results = yolo_model(image)

    rois = []

    for box in results[0].boxes.xyxy:
        x1, y1, x2, y2 = map(int, box)
        roi = image.crop((x1, y1, x2, y2))
        rois.append(roi)

    # ---- If no detection → fallback to full image ----
    if not rois:
        rois = [image]

    best_conf = 0
    best_result = None
    best_part = None

    # ---- STEP 2: Analyze each ROI ----
    for roi in rois:

        img_tensor = transform(roi).unsqueeze(0).to(DEVICE)

        # Body part
        with torch.no_grad():
            output = body_model(img_tensor)
            probs = torch.softmax(output, dim=1)[0]
            _, pred = torch.max(probs, 0)

        part = BODY_CLASSES[pred.item()]

        # Fracture detection
        model = fracture_models[part]

        with torch.no_grad():
            output = model(img_tensor)
            f_probs = torch.softmax(output, dim=1)[0]
            conf, f_pred = torch.max(f_probs, 0)

        if conf.item() > best_conf:
            best_conf = conf.item()
            best_result = FRACTURE_CLASSES[f_pred.item()]
            best_part = part

    # ---- FINAL OUTPUT ----
    print("\n🏥 HYBRID YOLO FRACTURE ANALYSIS")
    print("Most Likely Body Part :", best_part.upper())
    print("Diagnosis             :", best_result.upper())
    print(f"Confidence            : {best_conf:.2%}")

    if best_conf < 0.65:
        print("⚠️ Low confidence — review manually")

# =====================================================

if __name__ == "__main__":
    path = input("Enter path to X-ray image: ").strip()
    diagnose(path)
