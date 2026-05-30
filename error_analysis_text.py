import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import os
from torch.utils.data import DataLoader
from torchvision import transforms

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
        print(f"⚠️ Thất bại: Không tìm thấy file trọng số tại {HYBRID_WEIGHTS}")
        exit()

    # Bộ tiền xử lý chuẩn dữ liệu
    transform_val = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    # Đọc file CSV gốc để lấy ID của ảnh phục vụ việc truy vết failed cases
    df_val_origin = pd.read_csv(VAL_CSV)
    img_ids = df_val_origin['isic_id'].values
    
    val_loader = DataLoader(
        SkinDataset(VAL_CSV, DATA_DIR, transform_val), 
        batch_size=64, shuffle=False, num_workers=4, pin_memory=True
    )
    
    # Nạp mô hình Hybrid đề xuất
    model = HybridSkinModel(num_classes=8)
    model.load_state_dict(torch.load(HYBRID_WEIGHTS, map_location=device))
    model = model.to(device)
    model.eval()
    
    all_preds = []
    all_labels = []
    
    print("🔄 Đang chạy phân tích lỗi và thu thập các mẫu phân loại sai (Misclassified Samples)...")
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            with torch.amp.autocast('cuda'):
                outputs = model(images)
            _, preds = outputs.max(1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
    # =========================================================================
    # 2. SUB-TASK: THU THẬP VÀ VISUALIZATION FAILED CASES (DẠNG TEXT TABLE)
    # =========================================================================
    failed_samples = []
    error_counts_per_class = {c: 0 for c in CLASS_NAMES}
    total_samples = len(all_labels)
    
    for idx, (true_lbl, pred_lbl) in enumerate(zip(all_labels, all_preds)):
        if true_lbl != pred_lbl:
            true_name = CLASS_NAMES[true_lbl]
            pred_name = CLASS_NAMES[pred_lbl]
            
            # Tăng biến đếm lỗi của lớp thực tế
            error_counts_per_class[true_name] += 1
            
            # Thu thập thông tin chi tiết từng ca lỗi phục vụ truy vết
            failed_samples.append({
                'Index_Trong_Tap_Val': idx,
                'ISIC_ID_Anh_Loi': img_ids[idx],
                'Nhan_Thuc_Te (True Label)': true_name,
                'Mo_Hinh_Doan_Sai (Predicted Label)': pred_name
            })
            
    # Tạo bảng visualization thống kê lỗi tổng quan bằng văn bản
    df_failed_detail = pd.DataFrame(failed_samples)
    
    print("\n" + "="*90)
    print("📊 BẢNG THỐNG KÊ SỐ LƯỢNG MẪU BỊ PHÂN LOẠI SAI TRÊN TỪNG LỚP BỆNH (FAILED CASES)")
    print("="*90)
    for cls_name, count in error_counts_per_class.items():
        total_class_samples = np.sum(np.array(all_labels) == CLASS_NAMES.index(cls_name))
        err_rate = (count / total_class_samples * 100) if total_class_samples > 0 else 0
        print(f" ❖ Lớp bệnh {cls_name:<6} | Số ca đoán sai: {count:<4} / {total_class_samples:<4} mẫu | Tỉ lệ lỗi của lớp: {err_rate:.2f}%")
    print("="*90)
    print(f" 🌟 Tổng số mẫu bị lỗi toàn hệ thống: {len(failed_samples)} / {total_samples} mẫu.")
    print("="*90)
    
    # Hiển thị demo 5 ca lỗi đầu tiên ngay trên terminal để bạn kiểm tra cấu trúc
    print("\n📝 DEMO TRÍCH XUẤT 5 MẪU BỊ LỖI ĐẦU TIÊN ĐỂ PHỤC VỤ TRUY VẾT:")
    print(df_failed_detail.head(5).to_string(index=False))
    print("-" * 90)

    # =========================================================================
    # 3. XUẤT FILE DATA PHỤC VỤ PHẦN DISCUSSION CỦA BÁO CÁO
    # =========================================================================
    output_csv = r"C:\Users\HUYPHAT_PC\Documents\AI\SkinCancer-DeepLearning-ISIC\evaluation_results\misclassified_samples_report.csv"
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    df_failed_detail.to_csv(output_csv, index=False)
    print(f"✅ Đã thu thập và lưu danh sách {len(failed_samples)} mẫu lỗi tại: {output_csv}\n")