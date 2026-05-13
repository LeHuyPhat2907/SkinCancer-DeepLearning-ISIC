import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import os
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from tqdm import tqdm

# ==========================================
# 1. CẤU HÌNH ĐƯỜNG DẪN (Phát kiểm tra lại nhé)
# ==========================================
DATA_DIR = r'D:\SkinCancer_AI_ISIC\data\final_cleaned'
TRAIN_CSV = r'D:\SkinCancer_AI_ISIC\data\train.csv'
VAL_CSV = r'D:\SkinCancer_AI_ISIC\data\val.csv'
SAVE_PATH = r'D:\SkinCancer_AI_ISIC\models\best_baseline.pth'

# Tạo folder models nếu chưa có
os.makedirs(os.path.dirname(SAVE_PATH), exist_ok=True)

# ==========================================
# 2. ĐỊNH NGHĨA DATASET
# ==========================================
class SkinDataset(Dataset):
    def __init__(self, csv_file, img_dir, transform=None):
        self.data = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.transform = transform
        # Map label string sang số nếu cần (Giả sử csv đã có cột 'label_idx')
        self.labels = self.data['label_idx'].values 
        self.img_ids = self.data['isic_id'].values

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, f"{self.img_ids[idx]}.jpg")
        image = Image.open(img_path).convert('RGB')
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        
        if self.transform:
            image = self.transform(image)
        return image, label

# ==========================================
# 3. KIẾN TRÚC BASELINE CNN
# ==========================================
class SkinBaselineCNN(nn.Module):
    def __init__(self, num_classes=8):
        super(SkinBaselineCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2, 2)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 28 * 28, 512), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        return self.classifier(self.features(x))

# ==========================================
# 4. HÀM TRAIN & VALIDATE
# ==========================================
def train_one_epoch(model, loader, criterion, optimizer, scaler, device, epoch):
    model.train()
    running_loss, correct, total = 0, 0, 0
    pbar = tqdm(loader, desc=f"Epoch {epoch+1} [Train]")
    
    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        
        with torch.amp.autocast('cuda'): # Sửa lỗi scaler bản mới
            outputs = model(images)
            loss = criterion(outputs, labels)
        
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        pbar.set_postfix({'loss': f'{loss.item():.4f}', 'acc': f'{100.*correct/total:.2f}%'})
    return running_loss/len(loader), 100.*correct/total

def validate(model, loader, criterion, device):
    model.eval()
    running_loss, correct, total = 0, 0, 0
    with torch.no_grad():
        for images, labels in tqdm(loader, desc="[Valid]"):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    return running_loss/len(loader), 100.*correct/total

# ==========================================
# 5. CHƯƠNG TRÌNH CHÍNH (MAIN)
# ==========================================
if __name__ == "__main__":
    # Thiết lập thiết bị
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Chiến đấu trên: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")

    # Transform ảnh
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # Dataloader (CMP 40HX mạnh nên để num_workers=4 hoặc 8)
    train_ds = SkinDataset(TRAIN_CSV, DATA_DIR, transform)
    val_ds = SkinDataset(VAL_CSV, DATA_DIR, transform)
    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_ds, batch_size=64, shuffle=False, num_workers=4)

    # Khởi tạo mạng & Loss (Weighted)
    model = SkinBaselineCNN(num_classes=8).to(device)
    # Phát điền lại đúng mảng weights từ EDA của bạn vào đây nhé
    weights = torch.tensor([0.1, 0.1, 1.0, 0.5, 0.05, 1.0, 0.2, 1.5]).to(device) 
    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = optim.AdamW(model.parameters(), lr=1e-4)
    scaler = torch.amp.GradScaler('cuda')
    
    best_acc = 0
    for epoch in range(30):
        t_loss, t_acc = train_one_epoch(model, train_loader, criterion, optimizer, scaler, device, epoch)
        v_loss, v_acc = validate(model, val_loader, criterion, device)
        
        print(f"✅ Kết quả Epoch {epoch+1}: Val Acc: {v_acc:.2f}% | Val Loss: {v_loss:.4f}")
        
        # Lưu mô hình tốt nhất
        if v_acc > best_acc:
            best_acc = v_acc
            torch.save(model.state_dict(), SAVE_PATH)
            print(f"⭐ Đã lưu model tốt nhất với Acc: {v_acc:.2f}%")