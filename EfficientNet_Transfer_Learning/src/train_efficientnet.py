import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader
from torchvision import transforms

from dataset import SkinDataset
from efficientnet_model import EfficientNetTransfer
from logger import Logger

# ==================================================
# LOG SYSTEM
# ==================================================
sys.stdout = Logger("models/train_log.txt")

# ==================================================
# CREATE FOLDER
# ==================================================
os.makedirs("models", exist_ok=True)

# ==================================================
# PATHS
# ==================================================
TRAIN_CSV = "data/train.csv"
VAL_CSV = "data/val.csv"
IMAGE_DIR = "data/final_cleaned"

# ==================================================
# HYPERPARAMETERS
# ==================================================
BATCH_SIZE = 32
EPOCHS = 10
LEARNING_RATE = 1e-4
NUM_CLASSES = 8

# ==================================================
# DEVICE (🔥 GPU ENABLE)
# ==================================================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[DEVICE] Using: {DEVICE}")

if DEVICE.type == "cuda":
    print(f"[GPU] {torch.cuda.get_device_name(0)}")

# ==================================================
# TRANSFORMS
# ==================================================
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(0.2, 0.2, 0.2, 0.1),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# ==================================================
# DATASET (❌ bỏ spam log trong dataset)
# ==================================================
train_dataset = SkinDataset(TRAIN_CSV, IMAGE_DIR, train_transform)
val_dataset = SkinDataset(VAL_CSV, IMAGE_DIR, val_transform)

# ==================================================
# DATALOADER (🔥 SPEED BOOST)
# ==================================================
train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=4,        # 🔥 tăng tốc load
    pin_memory=True       # 🔥 GPU transfer faster
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=2,
    pin_memory=True
)

# ==================================================
# MODEL
# ==================================================
print("[MODEL] Initializing EfficientNet...")
model = EfficientNetTransfer(NUM_CLASSES).to(DEVICE)

# ==================================================
# LOSS + OPTIMIZER
# ==================================================
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# ==================================================
# CHECKPOINT
# ==================================================
best_val_acc = 0
start_epoch = 0
checkpoint_path = "models/checkpoint.pth"

if os.path.exists(checkpoint_path):
    print("[CHECKPOINT] Loading...")
    checkpoint = torch.load(checkpoint_path, map_location=DEVICE)

    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    start_epoch = checkpoint["epoch"] + 1
    best_val_acc = checkpoint["best_val_acc"]

    print(f"[CHECKPOINT] Resume from epoch {start_epoch}")

# ==================================================
# EPOCH LOG FILE
# ==================================================
epoch_log = open("models/epoch_log.txt", "a", encoding="utf-8")

# ==================================================
# TRAIN LOOP
# ==================================================
if __name__ == "__main__":

    for epoch in range(start_epoch, EPOCHS):

        print(f"\n🔥 ===== EPOCH {epoch+1}/{EPOCHS} START =====")

        # ================= TRAIN =================
        model.train()
        running_loss = 0
        correct = 0
        total = 0

        for i, (images, labels) in enumerate(train_loader):

            images = images.to(DEVICE, non_blocking=True)
            labels = labels.to(DEVICE, non_blocking=True)

            optimizer.zero_grad()

            outputs = model(images)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            # 🔥 nhẹ log batch (không spam)
            if i % 100 == 0:
                print(f"[TRAIN] Batch {i}/{len(train_loader)} | Loss={loss.item():.4f}")

        train_loss = running_loss / len(train_loader)
        train_acc = 100 * correct / total

        # ================= VALIDATION =================
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for images, labels in val_loader:

                images = images.to(DEVICE, non_blocking=True)
                labels = labels.to(DEVICE, non_blocking=True)

                outputs = model(images)
                loss = criterion(outputs, labels)

                val_loss += loss.item()

                _, predicted = torch.max(outputs, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()

        val_loss /= len(val_loader)
        val_acc = 100 * val_correct / val_total

        # ================= PRINT =================
        print(f"\n📊 Epoch {epoch+1}/{EPOCHS} DONE")
        print(f"Train Loss: {train_loss:.4f}")
        print(f"Train Acc : {train_acc:.2f}%")
        print(f"Val Loss  : {val_loss:.4f}")
        print(f"Val Acc   : {val_acc:.2f}%")
        print("-" * 50)

        # ================= SAVE LOG =================
        epoch_log.write(
            f"Epoch {epoch+1} | TL {train_loss:.4f} | TA {train_acc:.2f}% | "
            f"VL {val_loss:.4f} | VA {val_acc:.2f}%\n"
        )
        epoch_log.flush()

        # ================= SAVE BEST =================
        if val_acc > best_val_acc:
            best_val_acc = val_acc

            torch.save(model.state_dict(), "models/best_efficientnet.pth")

            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_val_acc": best_val_acc
            }, checkpoint_path)

            print("💾 Best model saved!")

        if DEVICE.type == "cuda":
            torch.cuda.empty_cache()

    epoch_log.close()

    print("\n✅ TRAINING COMPLETED!")
    print(f"🏆 Best Accuracy: {best_val_acc:.2f}%")