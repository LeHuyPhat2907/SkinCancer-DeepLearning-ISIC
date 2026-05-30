import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
import timm  # Sử dụng timm để gọi Vision Transformer chuẩn của Google

# Import lớp Dataset từ file gốc của bạn để đảm bảo đồng bộ luồng đọc ảnh
from train_baseline import SkinDataset

# =========================================================================
# 0. HỆ THỐNG TỰ ĐỘNG GHI LOG TERMINAL (LƯU TEXT LOG CHO PURE VIT)
# =========================================================================
class PureViTLogger(object):
    def __init__(self, filename="C:\\Users\\HUYPHAT_PC\\Documents\\AI\\SkinCancer-DeepLearning-ISIC\\models\\training_pure_vit.log"):
        self.terminal = sys.stdout
        self.log = open(filename, "w", encoding="utf-8")
        
    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()
        
    def flush(self):
        pass

# Tự động khởi tạo thư mục lưu trữ nếu chưa có
os.makedirs(r"C:\Users\HUYPHAT_PC\Documents\AI\SkinCancer-DeepLearning-ISIC\models", exist_ok=True)
sys.stdout = PureViTLogger() # Kích hoạt Logger ghi đè toàn bộ tiến trình

# =========================================================================
# 1. CẤU HÌNH ĐƯỜNG DẪN VÀ SIÊU THAM SỐ ĐỒNG BỘ
# =========================================================================
DATA_DIR = r'C:\Users\HUYPHAT_PC\Documents\AI\SkinCancer-DeepLearning-ISIC\data\final_cleaned'
TRAIN_CSV = r'C:\Users\HUYPHAT_PC\Documents\AI\SkinCancer-DeepLearning-ISIC\data\train.csv'
VAL_CSV = r'C:\Users\HUYPHAT_PC\Documents\AI\SkinCancer-DeepLearning-ISIC\data\val.csv'
MODEL_SAVE_PATH = r'C:\Users\HUYPHAT_PC\Documents\AI\SkinCancer-DeepLearning-ISIC\models\best_pure_vit.pth'
PLOT_SAVE_PATH = r'C:\Users\HUYPHAT_PC\Documents\AI\SkinCancer-DeepLearning-ISIC\models\pure_vit_learning_curves.png'

BATCH_SIZE = 128  # Giữ nguyên để đảm bảo tính công bằng khi đối sánh thực nghiệm
EPOCHS = 20      
LR = 5e-5        # Giữ nguyên mức LR tối ưu cho các họ mạng Attention nhằm tránh nổ gradient

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=========================================================================")
    print(f"🚀 TIẾN TRÌNH: HUẤN LUYỆN MÔ HÌNH PURE ViT (GOOGLE) TRÊN THIẾT BỊ: {device}")
    print("=========================================================================")

    # Thiết lập bộ tiền xử lý ảnh theo đúng chuẩn đầu vào của Vision Transformer
    transform_train = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    transform_val = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # Luồng DataLoader bốc dỡ dữ liệu song song qua VRAM
    train_loader = DataLoader(SkinDataset(TRAIN_CSV, DATA_DIR, transform_train), batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(SkinDataset(VAL_CSV, DATA_DIR, transform_val), batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

    # Khởi tạo mô hình Pure ViT nguyên bản (Sử dụng kiến trúc patch16 kích thước 224x224)
    print("🔄 Đang nạp cấu hình mạng và trọng số Pre-trained ImageNet của Pure ViT...")
    model = timm.create_model('vit_tiny_patch16_224', pretrained=True, num_classes=8)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=0.05)
    scaler = torch.amp.GradScaler('cuda') # Tăng tốc tính toán phần cứng qua Mixed Precision

    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    best_val_acc = 0.0

    for epoch in range(EPOCHS):
        # -----------------------------------------------------------------
        # GIAI ĐOẠN HUẤN LUYỆN (TRAINING PHASE)
        # -----------------------------------------------------------------
        model.train()
        train_loss, correct, total = 0.0, 0, 0
        pbar = tqdm(train_loader, desc=f"Pure ViT Epoch {epoch+1}/{EPOCHS} [Train]")
        
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

        # In kết quả Epoch đồng thời xuất thẳng vào file .log
        print(f"▶️ Kết quả Epoch {epoch+1}: Train Loss: {epoch_train_loss:.4f} | Train Acc: {epoch_train_acc:.2f}% || Val Loss: {epoch_val_loss:.4f} | Val Acc: {epoch_val_acc:.2f}%")

        history['train_loss'].append(epoch_train_loss)
        history['train_acc'].append(epoch_train_acc)
        history['val_loss'].append(epoch_val_loss)
        history['val_acc'].append(epoch_val_acc)

        # -----------------------------------------------------------------
        # TỰ ĐỘNG XUẤT ĐỒ THỊ VÀ CẬP NHẬT THEO THỜI GIAN THỰC
        # -----------------------------------------------------------------
        plt.figure(figsize=(12, 5))
        
        # Biểu diễn đường cong Loss
        plt.subplot(1, 2, 1)
        plt.plot(range(1, len(history['train_loss'])+1), history['train_loss'], label='Train Loss', color='#2ecc71', linewidth=2)
        plt.plot(range(1, len(history['val_loss'])+1), history['val_loss'], label='Val Loss', color='#e74c3c', linewidth=2)
        plt.title('Pure ViT Loss Curve', fontsize=12, fontweight='bold')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.legend()
        
        # Biểu diễn đường cong Accuracy
        plt.subplot(1, 2, 2)
        plt.plot(range(1, len(history['train_acc'])+1), history['train_acc'], label='Train Acc', color='#2ecc71', linewidth=2)
        plt.plot(range(1, len(history['val_acc'])+1), history['val_acc'], label='Val Acc', color='#e74c3c', linewidth=2)
        plt.title('Pure ViT Accuracy Curve', fontsize=12, fontweight='bold')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy (%)')
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(PLOT_SAVE_PATH, dpi=150)
        plt.close() # Giải phóng bộ nhớ đệm vẽ ảnh của CPU

        # Khấu trừ lưu trữ trọng số tối ưu nhất
        if epoch_val_acc > best_val_acc:
            best_val_acc = epoch_val_acc
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            print(f"⭐ Đã phát hiện đỉnh mới! Lưu trọng số Pure ViT với tập Val Acc: {best_val_acc:.2f}%")

    print("\n=========================================================================")
    print("✅ TIẾN TRÌNH HUẤN LUYỆN MÔ HÌNH PURE ViT ĐÃ HOÀN TẤT TRỌN VẸN!")
    print(f"📁 Nhật ký log văn bản lưu tại: C:\\Users\\HUYPHAT_PC\\Documents\\AI\\SkinCancer-DeepLearning-ISIC\\models\\training_pure_vit.log")
    print(f"📁 Hình ảnh đồ thị kiểm soát lưu tại: {PLOT_SAVE_PATH}")
    print(f"📁 Trọng số tối ưu lưu tại: {MODEL_SAVE_PATH}")
    print("=========================================================================\n")