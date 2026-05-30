import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import os
from torch.utils.data import DataLoader
from torchvision import transforms
from sklearn.metrics import confusion_matrix

# Import Dataset và cấu trúc mạng Lai từ dự án của Phát
from train_baseline import SkinDataset
from train_hybrid import HybridSkinModel

# =========================================================================
# 1. CẤU HÌNH ĐƯỜNG DẪN DỮ LIỆU VÀ TRỌNG SỐ MÔ HÌNH CHÍNH
# =========================================================================
DATA_DIR = r'C:\Users\HUYPHAT_PC\Documents\AI\SkinCancer-DeepLearning-ISIC\data\final_cleaned'
VAL_CSV = r'C:\Users\HUYPHAT_PC\Documents\AI\SkinCancer-DeepLearning-ISIC\data\val.csv'
HYBRID_WEIGHTS = r'C:\Users\HUYPHAT_PC\Documents\AI\SkinCancer-DeepLearning-ISIC\models\best_hybrid_model.pth'

CLASS_NAMES = ['MEL', 'NV', 'BCC', 'AK', 'BKL', 'DF', 'VASC', 'SCC']

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"⚙️ Thiết bị xử lý hiện tại: {device}")
    
    if not os.path.exists(HYBRID_WEIGHTS):
        print(f"⚠️ Thất bại: Không tìm thấy file trọng số mạng lai tại {HYBRID_WEIGHTS}")
        exit()

    # Bộ tiền xử lý chuẩn dữ liệu
    transform_val = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    val_loader = DataLoader(
        SkinDataset(VAL_CSV, DATA_DIR, transform_val), 
        batch_size=64, shuffle=False, num_workers=4, pin_memory=True
    )
    
    # Nạp mô hình Lai đề xuất
    print("🔄 Đang nạp mô hình Hybrid chính để bốc tách Ma trận nhầm lẫn...")
    model = HybridSkinModel(num_classes=8)
    model.load_state_dict(torch.load(HYBRID_WEIGHTS, map_location=device))
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
            
    all_labels = np.array(all_labels)
    all_preds = np.array(all_preds)

    # =========================================================================
    # 2. SUB-TASK: GENERATE CONFUSION MATRIX & VISUALIZATION HEATMAP (TEXT)
    # =========================================================================
    cm = confusion_matrix(all_labels, all_preds)
    
    # Chuyển ma trận sang định dạng DataFrame có gán nhãn hàng/cột để tạo hiệu ứng Heatmap văn bản
    df_cm = pd.DataFrame(cm, index=[f"Actual_{c}" for c in CLASS_NAMES], columns=[f"Pred_{c}" for c in CLASS_NAMES])
    
    print("\n" + "="*90)
    print("📊 MA TRẬN NHẦM LẪN (CONFUSION MATRIX) - MÔ HÌNH HYBRID CHÍNH ĐỀ TÀI")
    print("="*90)
    print(df_cm.to_string()) # In toàn bộ ma trận số liệu đều tăm tắp ra màn hình terminal
    print("="*90)
    
    # =========================================================================
    # 3. SUB-TASK: IDENTIFY CONFUSING CLASSES (TỰ ĐỘNG PHÂN TÍCH CẶP LỚP DỄ NHẦM LẪN)
    # =========================================================================
    print("\n🔍 KHẢO SÁT VÀ PHÂN TÍCH CÁC CẶP LỚP BỆNH DỄ GÂY NHẦM LẪN NHẤT:")
    print("-" * 60)
    
    confusing_list = []
    for i in range(len(CLASS_NAMES)):
        for j in range(len(CLASS_NAMES)):
            if i != j and cm[i, j] > 0: # Bỏ qua đường chéo chính (đoán đúng)
                confusing_list.append({
                    'actual': CLASS_NAMES[i],
                    'predicted': CLASS_NAMES[j],
                    'count': cm[i, j]
                })
                
    # Sắp xếp các cặp phân loại sai theo thứ tự giảm dần của số lượng mẫu bị nhầm
    confusing_list = sorted(confusing_list, key=lambda x: x['count'], reverse=True)
    
    # In ra top 3 cặp lớp khiến mô hình bối rối nhất
    print("⚠️ Top 3 cặp bệnh lý có tỷ lệ nhầm lẫn cao nhất của hệ thống:")
    for idx, item in enumerate(confusing_list[:3]):
        print(f"   {idx+1}. Thực tế là [{item['actual']}] nhưng mô hình đoán nhầm sang [{item['predicted']}]: {item['count']} ca.")
        
    print("-" * 60)

    # Tự động lưu cấu trúc ma trận nhầm lẫn này ra file CSV sạch để bạn import vào Excel tạo bảng
    output_csv = r"C:\Users\HUYPHAT_PC\Documents\AI\SkinCancer-DeepLearning-ISIC\evaluation_results\hybrid_confusion_matrix.csv"
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    df_cm.to_csv(output_csv)
    print(f"✅ Đã xuất cấu trúc số liệu ma trận nhầm lẫn thành công tại: {output_csv}\n")