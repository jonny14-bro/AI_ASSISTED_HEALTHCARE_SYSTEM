import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import sys

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# =====================================================
# 🔹 PATHS TO MODELS
# =====================================================

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
# 🔹 LOAD BODY PART MODEL
# =====================================================

body_model = models.densenet121(weights=None)
body_model.classifier = nn.Linear(1024, len(BODY_CLASSES))

state_dict = torch.load(BODY_MODEL_PATH, map_location=DEVICE, weights_only=True)
body_model.load_state_dict(state_dict)

body_model = body_model.to(DEVICE)
body_model.eval()

# =====================================================
# 🔹 LOAD FRACTURE MODELS
# =====================================================

fracture_models = {}

for part, path in FRACTURE_MODELS.items():
    model = models.densenet121(weights=None)
    model.classifier = nn.Linear(1024, 2)

    state_dict = torch.load(path, map_location=DEVICE, weights_only=True)
    model.load_state_dict(state_dict)

    model = model.to(DEVICE)
    model.eval()

    fracture_models[part] = model

# =====================================================
# 🔹 DIAGNOSIS FUNCTION
# =====================================================

def diagnose(image_path):

    image = Image.open(image_path).convert("RGB")
    img_tensor = transform(image).unsqueeze(0).to(DEVICE)

    # ---- Step 1: Body-part probabilities ----
    with torch.no_grad():
        output = body_model(img_tensor)
        probs = torch.softmax(output, dim=1)[0]

    # Get top 2 candidate parts
    topk = torch.topk(probs, k=2)

    candidates = [(BODY_CLASSES[i], probs[i].item()) for i in topk.indices]

    best_result = None
    best_conf = 0
    best_part = None

    # ---- Step 2: Evaluate each candidate model ----
    for part, part_prob in candidates:

        model = fracture_models[part]

        with torch.no_grad():
            output = model(img_tensor)
            f_probs = torch.softmax(output, dim=1)[0]
            conf, pred = torch.max(f_probs, 0)

        if conf.item() > best_conf:
            best_conf = conf.item()
            best_result = FRACTURE_CLASSES[pred.item()]
            best_part = part

    # ---- Final Report ----
    print("\n🏥 AI FRACTURE ANALYSIS")
    print("Most Likely Body Part :", best_part.upper())
    print("Diagnosis             :", best_result.upper())
    print(f"Confidence            : {best_conf:.2%}")

# =====================================================
# 🔹 ENTRY POINT
# =====================================================

if __name__ == "__main__":

    # Priority: command-line argument
    if len(sys.argv) > 1:
        image_path = sys.argv[1]

    # Otherwise ask interactively
    else:
        image_path = input("Enter path to X-ray image: ").strip()

    diagnose(image_path)
