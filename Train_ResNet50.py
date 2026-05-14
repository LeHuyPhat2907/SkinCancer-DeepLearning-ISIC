import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
import os

# Import lại class SkinDataset từ file hôm qua của bạn
from train_final import SkinDataset 

# ==========================================
# 1. CẤU HÌNH ĐƯỜNG DẪN
# ==========================================
DATA_DIR = r'D:\SkinCancer_AI_ISIC\data\final_cleaned'
TRAIN_CSV = r'D:\SkinCancer_AI_ISIC\data\train.csv'
VAL_CSV = r'D:\SkinCancer_AI_ISIC\data\val.csv'
SAVE_PATH = r'D:\SkinCancer_AI_ISIC\models\best_resnet50.pth'

BATCH_SIZE = 64
EPOCHS = 20 # ResNet50 hội tụ rất nhanh, 20 epoch là đủ thấy sự khác biệt
LR = 1e-4

# ==========================================
# 2. KHỞI TẠO RESNET50 (TRANSFER LEARNING)
# ==========================================
def get_resnet50(num_classes=8):
    # Load ResNet50 với trọng số Pre-trained từ ImageNet
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
    
    # Thay thế Classification Head
    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(in_features, 512),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(512, num_classes)
    )
    return model

# ==========================================
# 3. CHƯƠNG TRÌNH CHÍNH
# ==========================================
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Training ResNet50 trên: {device}")

    # Transform (Giữ nguyên chuẩn ImageNet)
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    train_ds = SkinDataset(TRAIN_CSV, DATA_DIR, transform)
    val_ds = SkinDataset(VAL_CSV, DATA_DIR, transform)
    
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    model = get_resnet50(num_classes=8).to(device)
    
    # Bạn có thể điền lại mảng weights từ EDA hôm qua vào đây
    criterion = nn.CrossEntropyLoss() 
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    scaler = torch.amp.GradScaler('cuda')

    best_acc = 0.0
    for epoch in range(EPOCHS):
        # --- TRAIN ---
        model.train()
        train_loss, correct, total = 0, 0, 0
        pbar = tqdm(train_loader, desc=f"ResNet Epoch {epoch+1}")
        for images, labels in pbar:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            with torch.amp.autocast('cuda'):
                outputs = model(images)
                loss = criterion(outputs, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            _, pred = outputs.max(1)
            total += labels.size(0)
            correct += pred.eq(labels).sum().item()
            pbar.set_postfix({'loss': f'{loss.item():.4f}', 'acc': f'{100.*correct/total:.2f}%'})

        # --- VALIDATE ---
        model.eval()
        val_correct, val_total = 0, 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, pred = outputs.max(1)
                val_total += labels.size(0)
                val_correct += pred.eq(labels).sum().item()
        
        v_acc = 100. * val_correct / val_total
        print(f"📊 Val Acc: {v_acc:.2f}%")

        if v_acc > best_acc:
            best_acc = v_acc
            torch.save(model.state_dict(), SAVE_PATH)
            print("⭐ Đã lưu ResNet50 tốt nhất!")

    print(f"✅ Xong! Best ResNet50 Acc: {best_acc:.2f}%")