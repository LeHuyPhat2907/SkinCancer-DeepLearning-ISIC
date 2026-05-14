import torch
import torch.nn as nn
from torchvision import models

def get_resnet50_model(num_classes=8):
    # Load ResNet50 với trọng số Pre-trained từ ImageNet
    # Sử dụng weights thay cho pretrained=True theo chuẩn PyTorch mới
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
    
    # Đóng băng các lớp đầu (tùy chọn - nếu muốn train nhanh hơn)
    # for param in model.parameters():
    #     param.requires_grad = False
    
    # Thay thế Classification Head (fc layer)
    # ResNet50 đầu ra của lớp cuối cùng là 2048
    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(in_features, 512),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(512, num_classes)
    )
    
    return model

# Khởi tạo kiểm tra
model_resnet = get_resnet50_model(num_classes=8)
print("✅ Đã load ResNet50 và thay đổi Classification Head!")