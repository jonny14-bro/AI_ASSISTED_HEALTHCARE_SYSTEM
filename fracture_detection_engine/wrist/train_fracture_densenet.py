import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms, datasets
from torch.utils.data import DataLoader
from tqdm import tqdm

# ================= CONFIG =================

DATA_DIR = r"D:\healthcare_assistant\fracture_detection_engine\data\train\fracture\wrist"
VAL_DIR  = r"D:\healthcare_assistant\fracture_detection_engine\data\val\fracture\wrist"

BATCH_SIZE = 8
EPOCHS = 15
LR = 1e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

SAVE_PATH = "wrist_fracture_densenet121.pth"

# ==========================================

# Transforms (safe for X-rays)
transform = transforms.Compose([
    transforms.Resize((320, 320)),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

val_transform = transforms.Compose([
    transforms.Resize((320, 320)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

# Load datasets
train_dataset = datasets.ImageFolder(DATA_DIR, transform=transform)
val_dataset   = datasets.ImageFolder(VAL_DIR, transform=val_transform)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader   = DataLoader(val_dataset, batch_size=BATCH_SIZE)

# ================= MODEL =================

# ✅ Load DenseNet121 WITH official pretrained weights
model = models.densenet121(weights="IMAGENET1K_V1")

# Replace classifier (2 classes: fracture / normal)
model.classifier = nn.Linear(1024, 2)

model = model.to(DEVICE)

# Freeze backbone (train classifier first)
for param in model.features.parameters():
    param.requires_grad = False

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.classifier.parameters(), lr=LR)

# ================= TRAIN LOOP =================

for epoch in range(EPOCHS):

    model.train()
    running_loss = 0

    for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}"):

        images, labels = images.to(DEVICE), labels.to(DEVICE)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    print(f"Train Loss: {running_loss/len(train_loader):.4f}")

    # ---------- Validation ----------
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)

            correct += (preds == labels).sum().item()
            total += labels.size(0)

    acc = correct / total
    print(f"Validation Accuracy: {acc:.4f}")

# Save model
torch.save(model.state_dict(), SAVE_PATH)

print("💀 Training Complete — Model Saved")
