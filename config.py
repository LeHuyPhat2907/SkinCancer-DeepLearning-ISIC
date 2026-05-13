import torch
from torchvision import transforms
from PIL import Image
import numpy as np

# 1. Định nghĩa pipeline chuẩn
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(), # Bước này tự động scale pixel về [0, 1]
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 2. Chạy thử trên 1 ảnh bất kỳ trong folder resized_224
img_path = r'd:\SkinCancer_AI_ISIC\data_processed\resized_224\ISIC_0000002.jpg' # Thay bằng file có thật
img = Image.open(img_path)
normalized_img = transform(img)

print(f"📏 Shape sau normalize: {normalized_img.shape}")
print(f"📊 Giá trị Min: {normalized_img.min().item():.4f}")
print(f"📊 Giá trị Max: {normalized_img.max().item():.4f}")