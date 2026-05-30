import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from PIL import Image
from tqdm import tqdm

# =========================================================================
# 0. HỆ THỐNG TỰ ĐỘNG GHI LOG TERMINAL (LƯU TEXT LOG TOÀN BỘ TIẾN TRÌNH)
# =========================================================================
class Logger(object):
    def __init__(self, filename="C:\\Users\\HUYPHAT_PC\\Documents\\AI\\SkinCancer-DeepLearning-ISIC\\models\\training_hybrid.log"):
        self.terminal = sys.stdout
        self.log = open(filename, "w", encoding="utf-8")
        
    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()
        
    def flush(self):
        pass

# Tạo thư mục lưu trữ mô hình và log nếu chưa tồn tại
os.makedirs(r"C:\Users\HUYPHAT_PC\Documents\AI\SkinCancer-DeepLearning-ISIC\models", exist_ok=True)
sys.stdout = Logger() # Bắt đầu ghi toàn bộ những gì print ra terminal vào file log

# =========================================================================
# 1. ĐỊNH NGHĨA LỚP DỮ LIỆU (SKIN DATASET)
# =========================================================================
class SkinDataset(Dataset):
    def __init__(self, csv_file, img_dir, transform=None):
        self.data = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.transform = transform
        self.labels = self.data['label'].values 
        self.img_ids = self.data['isic_id'].values
        
        # Đồng bộ nhãn tự động về khoảng [0, Num_Classes - 1] nếu nhãn bắt đầu từ 1
        if self.labels.min() == 1:
            self.labels = self.labels - 1
            print("ℹ️ Nhãn dữ liệu tự động chuyển dịch từ [1-8] về [0-7] để phù hợp với PyTorch.")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, f"{self.img_ids[idx]}.jpg")
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception:
            # Nếu ảnh lỗi, tạo ảnh trống kích thước 224x224 để không làm ngắt quãng luồng huấn luyện
            image = Image.new('RGB', (224, 224))
            
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        if self.transform:
            image = self.transform(image)
            
        return image, label

# =========================================================================
# 2. KIẾN TRÚC ATTENTION MODULE - CBAM
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
        x = x * self.ca(x) # Tính toán lọc thông tin theo Kênh (Channel Attention)
        x = x * self.sa(x) # Tính toán định vị không gian quan trọng (Spatial Attention)
        return x

# =========================================================================
# 3. KIẾN TRÚC MẠNG LAI CHÍNH: CNN (EFFICIENTNET) + CBAM + VIT (TRANSFORMER)
# =========================================================================
class HybridSkinModel(nn.Module):
    def __init__(self, num_classes=8, d_model=512, nhead=8, num_layers=4):
        super(HybridSkinModel, self).__init__()
        
        # Trích xuất đặc trưng cục bộ bằng EfficientNet-B0 Pretrained
        effnet = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
        self.backbone = effnet.features # Kích thước đầu ra cố định: [Batch, 1280, 7, 7]
        
        # Khối chú ý CBAM lọc đặc trưng nhiễu nền trên ảnh tế bào da
        self.cbam = CBAM(in_planes=1280, ratio=16)
        
        # Lớp Convolution 1x1 để ánh xạ số kênh từ 1280 về d_model (512) của Transformer
        self.conv_project = nn.Conv2d(1280, d_model, kernel_size=1)
        
        # Khối Transformer Encoder (Thành phần Vision Transformer)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=nhead, 
            dim_feedforward=d_model * 4, 
            dropout=0.3,
            activation='gelu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Token phân loại (Classification Token) tích lũy thông tin toàn cục
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        
        # Nhánh phân loại đầu ra (Classification Head)
        self.fc = nn.Sequential(
            nn.Linear(d_model, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        # 1. Trích xuất đặc trưng cục bộ thông qua CNN
        features = self.backbone(x)           # Output: [Batch, 1280, 7, 7]
        
        # 2. Tinh lọc vùng tổn thương qua CBAM Attention
        features = self.cbam(features)        # Output: [Batch, 1280, 7, 7]
        features = self.conv_project(features) # Output: [Batch, 512, 7, 7]
        
        # 3. Biến đổi không gian 7x7 thành chuỗi 49 Visual Tokens
        batch_size, C, H, W = features.shape
        features = features.view(batch_size, C, H * W).permute(0, 2, 1) # Output: [Batch, 49, 512]
        
        # 4. Ghép nối CLS Token vào chuỗi Token hình ảnh
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        tokens = torch.cat((cls_tokens, features), dim=1)               # Output: [Batch, 50, 512]
        
        # 5. Học ngữ cảnh tương quan toàn cục thông qua mạng Transformer
        transformer_out = self.transformer(tokens)                      # Output: [Batch, 50, 512]
        
        # 6. Trích xuất trạng thái của CLS Token và đưa vào bộ phân loại
        out_token = transformer_out[:, 0, :]
        return self.fc(out_token)

# =========================================================================
# 4. CHƯƠNG TRÌNH HUẤN LUYỆN CHÍNH (TRAINING LOOP & LIVE PLOTTING)
# =========================================================================
# Cấu hình các đường dẫn tuyệt đối đồng bộ dữ liệu trên ổ đĩa của bạn
DATA_DIR = r'C:\Users\HUYPHAT_PC\Documents\AI\SkinCancer-DeepLearning-ISIC\data\final_cleaned'
TRAIN_CSV = r'C:\Users\HUYPHAT_PC\Documents\AI\SkinCancer-DeepLearning-ISIC\data\train.csv'
VAL_CSV = r'C:\Users\HUYPHAT_PC\Documents\AI\SkinCancer-DeepLearning-ISIC\data\val.csv'
MODEL_SAVE_PATH = r'C:\Users\HUYPHAT_PC\Documents\AI\SkinCancer-DeepLearning-ISIC\models\best_hybrid_model.pth'
PLOT_SAVE_PATH = r'C:\Users\HUYPHAT_PC\Documents\AI\SkinCancer-DeepLearning-ISIC\models\hybrid_learning_curves.png'

BATCH_SIZE = 128
EPOCHS = 30
LR = 5e-5 # Learning rate nhỏ, tối ưu cho kiến trúc Transformer tránh nổ đạo hàm

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=========================================================================")
    print(f"🚀 HỆ THỐNG LAI [CNN + CBAM + ViT] ĐANG KHỞI ĐỘNG TRÊN THIẾT BỊ: {device}")
    print("=========================================================================")

    # Thiết lập bộ Tăng cường dữ liệu (Data Augmentation) mạnh hơn nhằm triệt tiêu Overfitting
    transform_train = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.1, contrast=0.1),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    transform_val = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # Khởi tạo DataLoader đọc luồng dữ liệu song song từ ổ cứng
    train_loader = DataLoader(SkinDataset(TRAIN_CSV, DATA_DIR, transform_train), batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(SkinDataset(VAL_CSV, DATA_DIR, transform_val), batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

    # Khởi tạo mô hình mạng lai 3 thành phần
    model = HybridSkinModel(num_classes=8).to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=0.05) # Áp dụng L2 Regularization mạnh mẽ
    scaler = torch.amp.GradScaler('cuda') # Sử dụng Mixed Precision để tối ưu hóa bộ nhớ và hiệu năng của GPU

    # Lịch sử lưu thông số vẽ đường cong học tập (Learning curves)
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    best_val_acc = 0.0

    for epoch in range(EPOCHS):
        # -----------------------------------------------------------------
        # GIAI ĐOẠN HUẤN LUYỆN (TRAINING PHASE)
        # -----------------------------------------------------------------
        model.train()
        train_loss, correct, total = 0.0, 0, 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Train]")
        
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

        # In kết quả Epoch ra màn hình và tự động đồng bộ vào file log .log
        print(f"▶️ Kết quả Epoch {epoch+1}: Train Loss: {epoch_train_loss:.4f} | Train Acc: {epoch_train_acc:.2f}% || Val Loss: {epoch_val_loss:.4f} | Val Acc: {epoch_val_acc:.2f}%")

        # Cập nhật lịch sử huấn luyện
        history['train_loss'].append(epoch_train_loss)
        history['train_acc'].append(epoch_train_acc)
        history['val_loss'].append(epoch_val_loss)
        history['val_acc'].append(epoch_val_acc)

        # -----------------------------------------------------------------
        # TỰ ĐỘNG VẼ ĐỒ THỊ LOSS VÀ ACCURACY LƯU RA FILE ẢNH
        # -----------------------------------------------------------------
        plt.figure(figsize=(12, 5))
        
        # Trực quan hóa Loss
        plt.subplot(1, 2, 1)
        plt.plot(range(1, len(history['train_loss'])+1), history['train_loss'], label='Train Loss', color='#e74c3c', linewidth=2)
        plt.plot(range(1, len(history['val_loss'])+1), history['val_loss'], label='Val Loss', color='#3498db', linewidth=2)
        plt.title('Loss Curve', fontsize=12, fontweight='bold')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.legend()
        
        # Trực quan hóa Accuracy
        plt.subplot(1, 2, 2)
        plt.plot(range(1, len(history['train_acc'])+1), history['train_acc'], label='Train Acc', color='#e74c3c', linewidth=2)
        plt.plot(range(1, len(history['val_acc'])+1), history['val_acc'], label='Val Acc', color='#3498db', linewidth=2)
        plt.title('Accuracy Curve', fontsize=12, fontweight='bold')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy (%)')
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(PLOT_SAVE_PATH, dpi=150) # Lưu đè cập nhật đồ thị theo thời gian thực
        plt.close() # Đóng luồng vẽ đồ thị giải phóng bộ nhớ RAM

        # Lưu lại checkpoint của mạng lai có độ chính xác tập kiểm thử cao nhất
        if epoch_val_acc > best_val_acc:
            best_val_acc = epoch_val_acc
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            print(f"⭐ Phát hiện mô hình tốt hơn! Đã lưu trọng số mạng lai mới với Val Acc đạt: {best_val_acc:.2f}%")

    print("\n=========================================================================")
    print("✅ QUÁ TRÌNH HUẤN LUYỆN MẠNG LAI ĐÃ HOÀN THÀNH XUẤT SẮC!")
    print(f"📁 Log văn bản chi tiết lưu tại: C:\\Users\\HUYPHAT_PC\\Documents\\AI\\SkinCancer-DeepLearning-ISIC\\models\\training_hybrid.log")
    print(f"📁 Đồ thị tiến trình huấn luyện cập nhật tại: {PLOT_SAVE_PATH}")
    print(f"📁 File trọng số tối ưu nhất lưu tại: {MODEL_SAVE_PATH}")
    print("=========================================================================")