# 1. Giới thiệu dự án (Project Overview)
Ghi ngắn gọn mục tiêu của đồ án.

Tên dự án: Skin Cancer Classification using Hybrid Deep Learning Model.

Mô tả: Hệ thống hỗ trợ chẩn đoán 8 loại tổn thương da từ bộ dữ liệu ISIC 2024, sử dụng kết hợp giữa CNN (EfficientNet) và Vision Transformer (ViT).

# 2. Cấu trúc thư mục (Project Structure)
├── data/               # Chứa file CSV (train, val, test)
├── models/             # Lưu trữ các file trọng số (.pth) sau khi train
├── notebooks/          # Các file EDA, thử nghiệm ban đầu
├── src/
│   ├── dataset.py      # Định nghĩa SkinDataset
│   ├── models.py       # Định nghĩa kiến trúc Baseline, ResNet, Hybrid
│   └── train.py        # Script huấn luyện chính
└── README.md

# 3. Nhật ký huấn luyện (Training Logs/Benchmark)
Model,Params,Val Accuracy,Status
Baseline CNN,51.4M,72.23%,Overfitting detected
ResNet50 (Transfer),23.5M,Đang cập nhật,Training...
Hybrid Model,-,-,Planned

# 4. Cấu hình hệ thống (System Environment)
Hardware: NVIDIA CMP 40HX (10GB VRAM).
Framework: PyTorch 2.x, Torchvision, Albumentations.
Dataset: ISIC 2024 (Skin Lesion Analysis) - ~51k images.

#############################################################
Link: Dataset: https://drive.google.com/drive/folders/1s2-MRwm4CM1GmB-KpIQRvzUiIzBUnSRI?usp=sharing

Link Toàn bộ mô hình đã Train: https://drive.google.com/drive/folders/14ktidAxIjNGVkswyBz7ntkK-ytiyA2NZ?usp=sharing







