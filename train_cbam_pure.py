import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
from torch.utils.data import DataLoader
from torchvision import models, transforms
from tqdm import tqdm

# Import lớp Dataset từ file gốc của bạn để đồng bộ luồng đọc dữ liệu ảnh
from train_baseline import SkinDataset

# =========================================================================
# 0. HỆ THỐNG TỰ ĐỘNG GHI LOG TERMINAL (LƯU TEXT LOG CHO MÔ HÌNH CBAM)
# =========================================================================
class CBAMLogger(object):
    def __init__(self, filename="C:\\Users\\HUYPHAT_PC\\Documents\\AI\\SkinCancer-DeepLearning-ISIC\\models\\training_pure_cbam.log"):
        self.terminal = sys.stdout
        self.log = open(filename, "w", encoding="utf-8")
        
    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()
        
    def flush(self):
        pass

# Tự động tạo thư mục lưu trữ nếu chưa tồn tại
os.makedirs(r"C:\Users\HUYPHAT_PC\Documents\AI\SkinCancer-DeepLearning-ISIC\models", exist_ok=True)
sys.stdout = CBAMLogger() # Kích hoạt Logger bắt trọn mọi thông tin hiển thị

# =========================================================================
# 1. ĐỊNH NGHĨA KHỐI CHÚ Ý CBAM (CHANNEL ATTENTION & SPATIAL ATTENTION)
# =========================================================================
class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
           
        self.fc = nn.Sequential(
            nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        return self.sigmoid(avg_out + max_out)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)

class CBAM(nn.Module):
    def __init__(self, in_planes, ratio=16, kernel_size=7):
        super(CBAM, self).__init__()
        self.ca = ChannelAttention(in_planes, ratio)
        self.sa = SpatialAttention(kernel_size)

    def forward(self, x):
        x = x * self.ca(x) # Bước 1: Lọc đặc trưng quan trọng theo Kênh (Channel)
        x = x * self.sa(x) # Bước 2: Khóa mục tiêu vùng tổn thương theo Không gian (Spatial)
        return x

# =========================================================================
# 2. KIẾN TRÚC MẠNG ĐỐI SÁNH: EFFICIENTNET-B0 + CBAM MODULE
# =========================================================================
class EfficientNetCBAMModel(nn.Module):
    def __init__(self, num_classes=8):
        super(EfficientNetCBAMModel, self).__init__()
        # Sử dụng EfficientNet-B0 làm Backbone giống hệt thành phần CNN của mạng lai
        effnet = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
        self.features = effnet.features # Đầu ra cố định: [Batch, 1280, 7, 7]
        
        # Tích hợp khối attention CBAM ngay sau đặc trưng tầng sâu của CNN
        self.cbam = CBAM(in_planes=1280, ratio=16)
        
        # Bộ phân loại đầu ra kết nối từ Feature Map sau Attention
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(1280, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.features(x)  # Trích xuất đặc trưng hình học cơ bản
        x = self.cbam(x)      # Đưa qua bộ lọc attention để làm nổi bật tế bào bệnh
        x = self.pool(x)      # nén không gian
        return self.classifier(x)

# =========================================================================
# 3. TIẾN TRÌNH HUẤN LUYỆN CHÍNH (TRAINING & LIVE PLOTTING)
# =========================================================================
DATA_DIR = r'C:\Users\HUYPHAT_PC\Documents\AI\SkinCancer-DeepLearning-ISIC\data\final_cleaned'
TRAIN_CSV = r'C:\Users\HUYPHAT_PC\Documents\AI\SkinCancer-DeepLearning-ISIC\data\train.csv'
VAL_CSV = r'C:\Users\HUYPHAT_PC\Documents\AI\SkinCancer-DeepLearning-ISIC\data\val.csv'
MODEL_SAVE_PATH = r'C:\Users\HUYPHAT_PC\Documents\AI\SkinCancer-DeepLearning-ISIC\models\best_efficientnet_cbam.pth'
PLOT_SAVE_PATH = r'C:\Users\HUYPHAT_PC\Documents\AI\SkinCancer-DeepLearning-ISIC\models\cbam_pure_learning_curves.png'

BATCH_SIZE = 128  # Đồng bộ tham số để đảm bảo tính thực nghiệm công bằng
EPOCHS = 20      
LR = 5e-5        

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=========================================================================")
    print(f"🚀 TIẾN TRÌNH: HUẤN LUYỆN MÔ HÌNH CNN + CBAM MODULE TRÊN THIẾT BỊ: {device}")
    print("=========================================================================")

    # Thiết lập bộ tiền xử lý tăng cường dữ liệu ảnh da liễu
    transform_train = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    transform_val = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    train_loader = DataLoader(SkinDataset(TRAIN_CSV, DATA_DIR, transform_train), batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(SkinDataset(VAL_CSV, DATA_DIR, transform_val), batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

    # Khởi tạo mô hình
    model = EfficientNetCBAMModel(num_classes=8).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=0.05)
    scaler = torch.amp.GradScaler('cuda') # Sử dụng Mixed Precision tăng tốc tối đa phần cứng

    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    best_val_acc = 0.0

    for epoch in range(EPOCHS):
        # -----------------------------------------------------------------
        # GIAI ĐOẠN HUẤN LUYỆN (TRAINING PHASE)
        # -----------------------------------------------------------------
        model.train()
        train_loss, correct, total = 0.0, 0, 0
        pbar = tqdm(train_loader, desc=f"CNN+CBAM Epoch {epoch+1}/{EPOCHS} [Train]")
        
        for images, labels in pbar:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            
            with torch.amp.autocast('cuda'):
                outputs = model(images)
                loss = criterion(outputs, labels)
                
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            train_loss += loss.item() * images.size(0)
            _, pred = outputs.max(1)
            total += labels.size(0)
            correct += pred.eq(labels).sum().item()
            pbar.set_postfix({'loss': f'{loss.item():.4f}', 'acc': f'{100.*correct/total:.2f}%'})

        epoch_train_loss = train_loss / total
        epoch_train_acc = 100. * correct / total

        # -----------------------------------------------------------------
        # GIAI ĐOẠN KIỂM THỬ (VALIDATION PHASE)
        # -----------------------------------------------------------------
        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0
        
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                with torch.amp.autocast('cuda'):
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                
                val_loss += loss.item() * images.size(0)
                _, pred = outputs.max(1)
                val_total += labels.size(0)
                val_correct += pred.eq(labels).sum().item()

        epoch_val_loss = val_loss / val_total
        epoch_val_acc = 100. * val_correct / val_total

        # Kết quả in ra terminal sẽ đồng bộ trực tiếp vào file .log nhờ bộ Logger tự chế
        print(f"▶️ Kết quả Epoch {epoch+1}: Train Loss: {epoch_train_loss:.4f} | Train Acc: {epoch_train_acc:.2f}% || Val Loss: {epoch_val_loss:.4f} | Val Acc: {epoch_val_acc:.2f}%")

        history['train_loss'].append(epoch_train_loss)
        history['train_acc'].append(epoch_train_acc)
        history['val_loss'].append(epoch_val_loss)
        history['val_acc'].append(epoch_val_acc)

        # -----------------------------------------------------------------
        # TỰ ĐỘNG XUẤT ĐỒ THỊ VÀ CẬP NHẬT THEO THỜI GIAN THỰC
        # -----------------------------------------------------------------
        plt.figure(figsize=(12, 5))
        
        # Biểu diễn đồ thị Loss
        plt.subplot(1, 2, 1)
        plt.plot(range(1, len(history['train_loss'])+1), history['train_loss'], label='Train Loss', color='#16a085', linewidth=2)
        plt.plot(range(1, len(history['val_loss'])+1), history['val_loss'], label='Val Loss', color='#d35400', linewidth=2)
        plt.title('CNN + CBAM Loss Curve', fontsize=12, fontweight='bold')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.legend()
        
        # Biểu diễn đồ thị Accuracy
        plt.subplot(1, 2, 2)
        plt.plot(range(1, len(history['train_acc'])+1), history['train_acc'], label='Train Acc', color='#16a085', linewidth=2)
        plt.plot(range(1, len(history['val_acc'])+1), history['val_acc'], label='Val Acc', color='#d35400', linewidth=2)
        plt.title('CNN + CBAM Accuracy Curve', fontsize=12, fontweight='bold')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy (%)')
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(PLOT_SAVE_PATH, dpi=150)
        plt.close() # Giải phóng tài nguyên RAM hệ thống sau khi lưu ảnh

        # Kiểm tra lưu checkpoint tốt nhất
        if epoch_val_acc > best_val_acc:
            best_val_acc = epoch_val_acc
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            print(f"⭐ Đã tìm thấy điểm tối ưu mới! Lưu trọng số với tập Val Acc đạt: {best_val_acc:.2f}%")

    print("\n=========================================================================")
    print("✅ TIẾN TRÌNH HUẤN LUYỆN MÔ HÌNH CNN + CBAM MODULE HOÀN THÀNH XUẤT SẮC!")
    print(f"📁 Nhật ký log văn bản lưu tại: C:\\Users\\HUYPHAT_PC\\Documents\\AI\\SkinCancer-DeepLearning-ISIC\\models\\training_pure_cbam.log")
    print(f"📁 Hình ảnh đồ thị kiểm soát lưu tại: {PLOT_SAVE_PATH}")
    print(f"📁 File trọng số tối ưu lưu tại: {MODEL_SAVE_PATH}")
    print("=========================================================================\n")