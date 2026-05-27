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

# Import lại class SkinDataset từ file gốc của bạn (Đảm bảo file train_final.py nằm cùng thư mục)
try:
    from train_final import SkinDataset
except ImportError:
    from train_baseline import SkinDataset

# =========================================================================
# 0. HỆ THỐNG TỰ ĐỘNG GHI LOG TERMINAL (LƯU TEXT LOG CHO RESNET50)
# =========================================================================
class ResNetLogger(object):
    def __init__(self, filename="D:\\SkinCancer_AI_ISIC\\models\\training_pure_resnet50.log"):
        self.terminal = sys.stdout
        self.log = open(filename, "w", encoding="utf-8")
        
    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()
        
    def flush(self):
        pass

# Tự động tạo thư mục lưu trữ nếu chưa tồn tại
os.makedirs(r"D:\SkinCancer_AI_ISIC\models", exist_ok=True)
sys.stdout = ResNetLogger() # Kích hoạt Logger bắt trọn mọi thông tin từ terminal

# =========================================================================
# 1. CẤU HÌNH ĐƯỜNG DẪN VÀ SIÊU THAM SỐ ĐỒNG BỘ THỰC NGHIỆM
# =========================================================================
DATA_DIR = r'D:\SkinCancer_AI_ISIC\data\final_cleaned'
TRAIN_CSV = r'D:\SkinCancer_AI_ISIC\data\train.csv'
VAL_CSV = r'D:\SkinCancer_AI_ISIC\data\val.csv'
MODEL_SAVE_PATH = r'D:\SkinCancer_AI_ISIC\models\best_resnet50.pth'
PLOT_SAVE_PATH = r'D:\SkinCancer_AI_ISIC\models\resnet50_pure_learning_curves.png'

BATCH_SIZE = 64  # Giữ nguyên để so sánh công bằng hiệu năng thực nghiệm giữa các kiến trúc
EPOCHS = 20      
LR = 1e-4        

# =========================================================================
# 2. KHỞI TẠO KIẾN TRÚC RESNET50 (TRANSFER LEARNING)
# =========================================================================
def get_resnet50(num_classes=8):
    # Load ResNet50 với trọng số Pre-trained từ ImageNet
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
    
    # Thay thế Classification Head (fc layer) kết nối chính xác với 8 lớp bài toán da liễu
    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(in_features, 512),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(512, num_classes)
    )
    return model

# =========================================================================
# 3. TIẾN TRÌNH HUẤN LUYỆN CHÍNH (TRAINING & LIVE PLOTTING)
# =========================================================================
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=========================================================================")
    print(f"🚀 TIẾN TRÌNH: HUẤN LUYỆN MÔ HÌNH PURE RESNET50 TRÊN THIẾT BỊ: {device}")
    print("=========================================================================")

    # Transform (Giữ nguyên chuẩn ImageNet và đồng bộ với thực nghiệm của bạn)
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # Khởi tạo DataLoader tối ưu hóa luồng nạp ảnh song song vào VRAM
    train_ds = SkinDataset(TRAIN_CSV, DATA_DIR, transform)
    val_ds = SkinDataset(VAL_CSV, DATA_DIR, transform)
    
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

    model = get_resnet50(num_classes=8).to(device)
    
    criterion = nn.CrossEntropyLoss() 
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    scaler = torch.amp.GradScaler('cuda') # Sử dụng Mixed Precision (AMP) để giải phóng tài nguyên phần cứng

    # Lịch sử lưu trữ thông số vẽ đường cong học tập
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    best_val_acc = 0.0

    for epoch in range(EPOCHS):
        # -----------------------------------------------------------------
        # GIAI ĐOẠN HUẤN LUYỆN (TRAINING PHASE)
        # -----------------------------------------------------------------
        model.train()
        train_loss, correct, total = 0.0, 0, 0
        pbar = tqdm(train_loader, desc=f"ResNet Epoch {epoch+1}/{EPOCHS} [Train]")
        
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

        # In kết quả Epoch đồng thời xuất thẳng vào file .log lưu trữ sạch sẽ
        print(f"▶️ Kết quả Epoch {epoch+1}: Train Loss: {epoch_train_loss:.4f} | Train Acc: {epoch_train_acc:.2f}% || Val Loss: {epoch_val_loss:.4f} | Val Acc: {epoch_val_acc:.2f}%")

        history['train_loss'].append(epoch_train_loss)
        history['train_acc'].append(epoch_train_acc)
        history['val_loss'].append(epoch_val_loss)
        history['val_acc'].append(epoch_val_acc)

        # -----------------------------------------------------------------
        # TỰ ĐỘNG XUẤT ĐỒ THỊ VÀ CẬP NHẬT THEO THỜI GIAN THỰC (.PNG)
        # -----------------------------------------------------------------
        plt.figure(figsize=(12, 5))
        
        # Đồ thị biểu diễn Loss curve
        plt.subplot(1, 2, 1)
        plt.plot(range(1, len(history['train_loss'])+1), history['train_loss'], label='Train Loss', color='#9b59b6', linewidth=2)
        plt.plot(range(1, len(history['val_loss'])+1), history['val_loss'], label='Val Loss', color='#e74c3c', linewidth=2)
        plt.title('ResNet50 Loss Curve', fontsize=12, fontweight='bold')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.legend()
        
        # Đồ thị biểu diễn Accuracy curve
        plt.subplot(1, 2, 2)
        plt.plot(range(1, len(history['train_acc'])+1), history['train_acc'], label='Train Acc', color='#9b59b6', linewidth=2)
        plt.plot(range(1, len(history['val_acc'])+1), history['val_acc'], label='Val Acc', color='#e74c3c', linewidth=2)
        plt.title('ResNet50 Accuracy Curve', fontsize=12, fontweight='bold')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy (%)')
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(PLOT_SAVE_PATH, dpi=150)
        plt.close() # Giải phóng bộ nhớ đệm CPU sau khi ghi đè dữ liệu ảnh thành công

        # Kiểm tra và đóng băng lưu trữ file trọng số tối ưu nhất
        if epoch_val_acc > best_val_acc:
            best_val_acc = epoch_val_acc
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            print(f"⭐ Phát hiện mô hình tốt hơn! Đã lưu trọng số ResNet50 với Val Acc đạt: {best_val_acc:.2f}%")

    print("\n=========================================================================")
    print("✅ TIẾN TRÌNH HUẤN LUYỆN MÔ HÌNH RESNET50 ĐÃ HOÀN TẤT TRỌN VẸN!")
    print(f"📁 Nhật ký log văn bản lưu tại: D:\\SkinCancer_AI_ISIC\\models\\training_pure_resnet50.log")
    print(f"📁 Hình ảnh đồ thị kiểm soát lưu tại: {PLOT_SAVE_PATH}")
    print(f"📁 File trọng số tối ưu lưu tại: {MODEL_SAVE_PATH}")
    print("=========================================================================\n")