import os
import cv2
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2

# ==========================================
# 1. ĐỊNH NGHĨA AUGMENTATION (Đã chốt ở các task trước)
# ==========================================
train_transform = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.Rotate(limit=45, p=0.5, border_mode=cv2.BORDER_CONSTANT, value=0),
    A.RandomBrightnessContrast(brightness_limit=0.15, contrast_limit=0.15, p=0.5),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2()
])

val_test_transform = A.Compose([
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2()
])

# ==========================================
# 2. CUSTOM DATASET CLASS
# ==========================================
class SkinCancerDataset(Dataset):
    def __init__(self, csv_file, img_dir, transform=None):
        self.df = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_id = self.df.iloc[idx]['isic_id']
        label = self.df.iloc[idx]['label']
        
        img_path = os.path.join(self.img_dir, f"{img_id}.jpg")
        image = cv2.imread(img_path)
        
        if image is None:
            raise FileNotFoundError(f"Không tìm thấy ảnh: {img_path}")
            
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.transform:
            augmented = self.transform(image=image)
            image = augmented['image']
            
        return image, torch.tensor(label, dtype=torch.long)

# ==========================================
# 3. KHỞI TẠO DATALOADERS
# ==========================================
if __name__ == '__main__':
    # Đường dẫn
    IMG_DIR = r'd:\SkinCancer_AI_ISIC\data_processed\final_cleaned'
    TRAIN_CSV = r'd:\SkinCancer_AI_ISIC\data\train.csv'
    VAL_CSV = r'd:\SkinCancer_AI_ISIC\data\val.csv'

    # Tạo Dataset
    train_ds = SkinCancerDataset(TRAIN_CSV, IMG_DIR, transform=train_transform)
    val_ds = SkinCancerDataset(VAL_CSV, IMG_DIR, transform=val_test_transform)

    # Tạo DataLoader (Tăng num_workers để load ảnh nhanh hơn)
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_ds, batch_size=32, shuffle=False, num_workers=4)

    # ==========================================
    # 4. TEST BATCH LOADING (Sub-task quan trọng nhất)
    # ==========================================
    images, labels = next(iter(train_loader))
    
    print("\n✅ DATALOADER PIPELINE ĐÃ SẴN SÀNG!")
    print(f"📦 Batch images shape: {images.shape}") # [32, 3, 224, 224]
    print(f"🏷️ Batch labels shape: {labels.shape}") # [32]
    print(f"🚀 Thiết bị: {'GPU' if torch.cuda.is_available() else 'CPU'}")