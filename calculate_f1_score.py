import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import os
from torch.utils.data import DataLoader
from torchvision import transforms
from sklearn.metrics import classification_report

# Import Dataset và cấu trúc mạng Lai từ dự án của Phát
from train_baseline import SkinDataset
from train_hybrid import HybridSkinModel
import timm

# =========================================================================
# 1. CẤU HÌNH ĐƯỜNG DẪN DỮ LIỆU VÀ TRỌNG SỐ
# =========================================================================
DATA_DIR = r'C:\Users\HUYPHAT_PC\Documents\AI\SkinCancer-DeepLearning-ISIC\data\final_cleaned'
VAL_CSV = r'C:\Users\HUYPHAT_PC\Documents\AI\SkinCancer-DeepLearning-ISIC\data\val.csv'

WEIGHTS_PATHS = {
    'Best_baseline': r"C:\Users\HUYPHAT_PC\Documents\AI\SkinCancer-DeepLearning-ISIC\model\best_baseline.pth",
    'Pure_EfficientNet' : r"C:\Users\HUYPHAT_PC\Documents\AI\SkinCancer-DeepLearning-ISIC\models\best_eff.pth",
    'Pure_EfficientNet': r'C:\Users\HUYPHAT_PC\Documents\AI\SkinCancer-DeepLearning-ISIC\models\best_pure_efficientnet.pth',
    'Pure_ViT': r'C:\Users\HUYPHAT_PC\Documents\AI\SkinCancer-DeepLearning-ISIC\models\best_pure_vit.pth',
    'Pure_DeiT': r'C:\Users\HUYPHAT_PC\Documents\AI\SkinCancer-DeepLearning-ISIC\models\best_pure_deit.pth',
    'Hybrid_Model': r'C:\Users\HUYPHAT_PC\Documents\AI\SkinCancer-DeepLearning-ISIC\models\best_hybrid_model.pth'
}

CLASS_NAMES = ['MEL', 'NV', 'BCC', 'AK', 'BKL', 'DF', 'VASC', 'SCC']

def load_model_by_name(model_name, device):
    if model_name == 'Pure_EfficientNet':
        from torchvision import models
        model = models.efficientnet_b0(weights=None)
        in_features = model.classifier[1].in_features
        model.classifier = nn.Sequential(nn.Dropout(p=0.3), nn.Linear(in_features, 8))
    elif model_name == 'Pure_ViT':
        model = timm.create_model('vit_tiny_patch16_224', pretrained=False, num_classes=8)
    elif model_name == 'Pure_DeiT':
        model = timm.create_model('deit_tiny_distilled_patch16_224', pretrained=False, num_classes=8)
    elif model_name == 'Hybrid_Model':
        model = HybridSkinModel(num_classes=8)
    return model

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"⚙️ Thiết bị tính toán hiện tại: {device}")
    
    transform_val = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    val_loader = DataLoader(
        SkinDataset(VAL_CSV, DATA_DIR, transform_val), 
        batch_size=64, shuffle=False, num_workers=4, pin_memory=True
    )
    
    f1_summary = {}

    for name, path in WEIGHTS_PATHS.items():
        if not os.path.exists(path):
            print(f"⚠️ Bỏ qua {name} (Không tìm thấy file trọng số)")
            continue
            
        print(f"🔄 Đang tính toán F1-Score cho mô hình: {name}...")
        model = load_model_by_name(name, device)
        model.load_state_dict(torch.load(path, map_location=device))
        model = model.to(device)
        model.eval()
        
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                with torch.amp.autocast('cuda'):
                    outputs = model(images)
                _, preds = outputs.max(1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                
        report = classification_report(all_labels, all_preds, target_names=CLASS_NAMES, output_dict=True)
        
        # Trích xuất giá trị F1-Score
        model_f1 = {}
        for cls in CLASS_NAMES:
            model_f1[cls] = round(report[cls]['f1-score'] * 100, 2)
        model_f1['Macro_F1'] = round(report['macro avg']['f1-score'] * 100, 2)
        
        f1_summary[name] = model_f1

    # =========================================================================
    # 3. IN BẢNG ĐỐI SÁNH MARSDOWN CHUẨN ĐỂ ĐƯA VÀO BÁO CÁO
    # =========================================================================
    print("\n" + "="*80)
    print("🏆 BẢNG ĐỐI SÁNH CHỈ SỐ F1-SCORE (%) GIỮA CÁC MÔ HÌNH")
    print("="*80)
    df = pd.DataFrame(f1_summary).transpose()
    print(df.to_markdown())
    print("="*80)
    
    output_csv = r"C:\Users\HUYPHAT_PC\Documents\AI\SkinCancer-DeepLearning-ISIC\evaluation_results\f1_comparison.csv"
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    df.to_csv(output_csv)
    print(f"✅ Đã lưu bảng so sánh F1-Score tại: {output_csv}")