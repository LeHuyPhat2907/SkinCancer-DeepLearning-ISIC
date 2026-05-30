import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import os
from torch.utils.data import DataLoader
from torchvision import transforms
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize

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
NUM_CLASSES = len(CLASS_NAMES)

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"⚙️ Thiết bị xử lý hiện tại: {device}")
    
    if not os.path.exists(HYBRID_WEIGHTS):
        print(f"⚠️ Thất bại: Không tìm thấy file trọng số mạng lai tại {HYBRID_WEIGHTS}")
        exit()

    # Bộ tiền xử lý chuẩn dữ liệu Validation
    transform_val = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    val_loader = DataLoader(
        SkinDataset(VAL_CSV, DATA_DIR, transform_val), 
        batch_size=64, shuffle=False, num_workers=4, pin_memory=True
    )
    
    # Nạp mô hình Hybrid chính đề tài
    print("🔄 Đang nạp mô hình Hybrid đề xuất để tính toán chỉ số Robustness (ROC-AUC)...")
    model = HybridSkinModel(num_classes=8)
    model.load_state_dict(torch.load(HYBRID_WEIGHTS, map_location=device))
    model = model.to(device)
    model.eval()
    
    all_labels = []
    all_probs = [] # Lưu xác suất dự đoán (Softmax) thay vì nhãn cứng
    
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            with torch.amp.autocast('cuda'):
                outputs = model(images)
                probs = torch.softmax(outputs, dim=1) # Chuyển Logits về phân phối xác suất
                
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)

    # Nhị phân hóa nhãn phục vụ bài toán đa lớp theo chiến lược One-vs-Rest
    y_true_binarized = label_binarize(all_labels, classes=list(range(NUM_CLASSES)))

    # =========================================================================
    # 2. SUB-TASK: COMPUTE ROC CURVE & AUC SCORE PER CLASS
    # =========================================================================
    print("\n" + "="*80)
    print("🏆 KẾT QUẢ ĐÁNH GIÁ CHỈ SỐ AUC CHI TIẾT TỪNG LỚP BỆNH (HYBRID MODEL)")
    print("="*80)
    
    auc_scores = {}
    roc_coordinates_list = []

    for i in range(NUM_CLASSES):
        # Tính toán FPR và TPR cho từng lớp
        fpr, tpr, thresholds = roc_curve(y_true_binarized[:, i], all_probs[:, i])
        score_auc = auc(fpr, tpr)
        
        auc_scores[CLASS_NAMES[i]] = round(score_auc, 4)
        print(f" ❖ Lớp bệnh {CLASS_NAMES[i]:<6} | Chỉ số AUC đạt: {score_auc:.4f}")
        
        # Lưu lại tọa độ để xuất file phục vụ sub-task trực quan hóa (Plot visualization)
        for f, t in zip(fpr, tpr):
            roc_coordinates_list.append({
                'Class': CLASS_NAMES[i],
                'FPR_FalsePositiveRate': f,
                'TPR_TruePositiveRate': t
            })

    # Tính Macro AUC (Trung bình cộng diện tích của cả 8 lớp)
    macro_auc = round(np.mean(list(auc_scores.values())), 4)
    print("-"*80)
    print(f" ⭐ CHỈ SỐ MACRO AUC TỔNG THỂ HỆ THỐNG: {macro_auc:.4f}")
    print("="*80)

    # =========================================================================
    # 3. TRÍCH XUẤT VÀ LƯU TRỮ SẠCH SẼ TOÀN BỘ DỮ LIỆU SỐ LIỆU
    # =========================================================================
    # 3.1 Lưu bảng chỉ số AUC
    df_auc = pd.DataFrame([auc_scores])
    df_auc['Macro_AUC'] = macro_auc
    df_auc.index = ['AUC_Score']
    
    output_auc_csv = r"C:\Users\HUYPHAT_PC\Documents\AI\SkinCancer-DeepLearning-ISIC\evaluation_results\hybrid_auc_scores.csv"
    os.makedirs(os.path.dirname(output_auc_csv), exist_ok=True)
    df_auc.to_csv(output_auc_csv)
    
    # 3.2 Lưu tọa độ đồ thị ROC sang file CSV
    df_roc_coords = pd.DataFrame(roc_coordinates_list)
    output_roc_csv = r"C:\Users\HUYPHAT_PC\Documents\AI\SkinCancer-DeepLearning-ISIC\evaluation_results\hybrid_roc_coordinates.csv"
    df_roc_coords.to_csv(output_roc_csv, index=False)
    
    print(f"✅ Đã xuất bảng điểm số AUC tại: {output_auc_csv}")
    print(f"✅ Đã trích xuất trọn gói tọa độ đồ thị ROC tại: {output_roc_csv}\n")