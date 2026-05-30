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
# 1. CẤU HÌNH ĐƯỜNG DẪN (Phát chỉ cần sửa ở đây)
# ==========================================
DATA_DIR = r'C:\Users\HUYPHAT_PC\Documents\AI\SkinCancer-DeepLearning-ISIC\data\final_cleaned'
TRAIN_CSV = r'C:\Users\HUYPHAT_PC\Documents\AI\SkinCancer-DeepLearning-ISIC\data\train.csv'
VAL_CSV = r'C:\Users\HUYPHAT_PC\Documents\AI\SkinCancer-DeepLearning-ISIC\data\val.csv'
SAVE_PATH = r'C:\Users\HUYPHAT_PC\Documents\AI\SkinCancer-DeepLearning-ISIC\models\best_baseline.pth'

# Cấu hình Hyperparameters
BATCH_SIZE = 64  # Có thể tăng lên 128 nếu CMP 40HX còn trống VRAM
EPOCHS = 30
LR = 1e-4

# Tạo folder lưu model
os.makedirs(os.path.dirname(SAVE_PATH), exist_ok=True)

# ==========================================
# 2. ĐỊNH NGHĨA DATASET (Tự động xử lý nhãn)
# ==========================================
class SkinDataset(Dataset):
    def __init__(self, csv_file, img_dir, transform=None):
        self.data = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.transform = transform
        
        # Lấy nhãn và kiểm tra dải giá trị
        self.labels = self.data['label'].values 
        self.img_ids = self.data['isic_id'].values
        
        # Nếu nhãn nhỏ nhất là 1, tự động trừ 1 cho toàn bộ để về dải 0-7
        if self.labels.min() == 1:
            self.labels = self.labels - 1
            print(f"ℹ️ Đã tự động chuyển nhãn 1-8 về 0-7 cho file: {os.path.basename(csv_file)}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, f"{self.img_ids[idx]}.jpg")
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception:
            # Nếu lỗi file ảnh, tạo ảnh đen để tránh crash (ít xảy ra)
            image = Image.new('RGB', (224, 224))
            
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
# 4. CHƯƠNG TRÌNH CHÍNH
# ==========================================
if __name__ == "__main__":
    # 4.1. Kiểm tra thiết bị
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "None"
    print(f"\n🚀 TRẠNG THÁI: Chiến đấu trên {device} ({gpu_name})")
    
    if device.type == 'cpu':
        print("⚠️ CẢNH BÁO: Không tìm thấy GPU. Hãy kiểm tra lại Driver hoặc PyTorch CUDA!")

    # 4.2. Chuẩn bị Dữ liệu
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    train_ds = SkinDataset(TRAIN_CSV, DATA_DIR, transform)
    val_ds = SkinDataset(VAL_CSV, DATA_DIR, transform)
    
    # num_workers=4 giúp nạp ảnh nhanh hơn từ ổ cứng
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    # 4.3. Khởi tạo Mô hình & Optimizer
    model = SkinBaselineCNN(num_classes=8).to(device)
    
    # Trọng số lớp (Phát có thể thay bằng mảng tính được từ EDA)
    class_weights = torch.tensor([1.0] * 8).to(device) 
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    scaler = torch.amp.GradScaler('cuda') if device.type == 'cuda' else None

    # 4.4. Vòng lặp Huấn luyện
    best_acc = 0.0
    for epoch in range(EPOCHS):
        # --- PHASE: TRAIN ---
        model.train()
        train_loss, correct, total = 0, 0, 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Train]")
        
        for images, labels in pbar:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            
            if scaler:
                with torch.amp.autocast('cuda'):
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
            
            train_loss += loss.item()
            _, pred = outputs.max(1)
            total += labels.size(0)
            correct += pred.eq(labels).sum().item()
            pbar.set_postfix({'loss': f'{loss.item():.4f}', 'acc': f'{100.*correct/total:.2f}%'})

        # --- PHASE: VALIDATE ---
        model.eval()
        val_loss, val_correct, val_total = 0, 0, 0
        with torch.no_grad():
            for images, labels in tqdm(val_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Valid]"):
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                v_loss = criterion(outputs, labels)
                
                val_loss += v_loss.item()
                _, pred = outputs.max(1)
                val_total += labels.size(0)
                val_correct += pred.eq(labels).sum().item()
        
        v_acc = 100. * val_correct / val_total
        print(f"📊 Kết quả: Val Loss: {val_loss/len(val_loader):.4f} | Val Acc: {v_acc:.2f}%")

        # Lưu model tốt nhất
        if v_acc > best_acc:
            best_acc = v_acc
            torch.save(model.state_dict(), SAVE_PATH)
            print(f"⭐ Đã lưu model tốt nhất!")

    print(f"\n✅ HOÀN THÀNH! Model tốt nhất đạt: {best_acc:.2f}% Accuracy.")