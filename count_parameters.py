import torch
import torch.nn as nn
import pandas as pd
import timm
from torchvision import models

# Import cấu trúc mạng lai chính đề tài của Phát
from train_hybrid import HybridSkinModel

# =========================================================================
# 1. ĐỊNH NGHĨA HÀM ĐẾM THAM SỐ CHUẨN PYTORCH
# =========================================================================
def count_parameters(model):
    # Tổng số lượng tham số trong mạng
    total_params = sum(p.numel() for p in model.parameters())
    # Số lượng tham số thực sự tham gia huấn luyện (được cập nhật gradient)
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    # Quy đổi ra đơn vị Triệu tham số (Millions) để đưa vào báo cáo cho đẹp
    return total_params, trainable_params

if __name__ == "__main__":
    print("=========================================================================")
    print("🧮 HỆ THỐNG ĐANG TÍNH TOÁN SỐ LƯỢNG THAM SỐ (MODEL PARAMETERS)")
    print("=========================================================================")
    
    param_results = {}

    # --- 1. MÔ HÌNH PURE EFFICIENTNET-B0 ---
    effnet = models.efficientnet_b0(weights=None)
    in_features = effnet.classifier[1].in_features
    effnet.classifier = nn.Sequential(nn.Dropout(p=0.3), nn.Linear(in_features, 8))
    
    t_p, tr_p = count_parameters(effnet)
    param_results['Pure_EfficientNet-B0'] = {"Total_Params": t_p, "Trainable_Params": tr_p}

    # --- 2. MÔ HÌNH PURE VIT (vit_tiny_patch16_224) ---
    pure_vit = timm.create_model('vit_tiny_patch16_224', pretrained=False, num_classes=8)
    t_p, tr_p = count_parameters(pure_vit)
    param_results['Pure_ViT_Tiny'] = {"Total_Params": t_p, "Trainable_Params": tr_p}

    # --- 3. MÔ HÌNH PURE DEIT (deit_tiny_distilled_patch16_224) ---
    pure_deit = timm.create_model('deit_tiny_distilled_patch16_224', pretrained=False, num_classes=8)
    t_p, tr_p = count_parameters(pure_deit)
    param_results['Pure_DeiT_Tiny'] = {"Total_Params": t_p, "Trainable_Params": tr_p}

    # --- 4. MÔ HÌNH CHÍNH ĐỀ TÀI: HYBRID MODEL (CNN + CBAM + ViT) ---
    hybrid_model = HybridSkinModel(num_classes=8)
    t_p, tr_p = count_parameters(hybrid_model)
    param_results['Hybrid_Model_Proposed'] = {"Total_Params": t_p, "Trainable_Params": tr_p}

    # =========================================================================
    # 2. XUẤT BẢNG TỔNG HỢP SO SÁNH DẠNG TEXT ĐẸP MẮT
    # =========================================================================
    rows = []
    for m_name, p_data in param_results.items():
        rows.append({
            "Mô hình": m_name,
            "Tổng số tham số (Gốc)": f"{p_data['Total_Params']:,}",
            "Số tham số huấn luyện": f"{p_data['Trainable_Params']:,}",
            "Kích thước tham số (M)": f"{p_data['Total_Params'] / 1e6:.2f} M"
        })
        
    df_params = pd.DataFrame(rows)
    print(df_params.to_markdown(index=False))
    print("=========================================================================")
    
    # Tự động xuất ra file CSV phục vụ đính kèm phụ lục báo cáo
    output_path = r"C:\Users\HUYPHAT_PC\Documents\AI\SkinCancer-DeepLearning-ISIC\evaluation_results\model_parameters_comparison.csv"
    df_params.to_csv(output_path, index=False, encoding='utf-8-sig')
    print(f"✅ Đã trích xuất và lưu bảng tham số trọn gói tại: {output_path}\n")