import torch
import torch.nn as nn
from torchvision import transforms, models
import timm
from PIL import Image
import streamlit as st
import pandas as pd
import numpy as np
import time
import os

# Import cấu trúc mạng Lai chính đề tài của Phát
from train_hybrid import HybridSkinModel

# Cấu hình thông tin giao diện Web chuẩn ứng dụng Y tế
st.set_page_config(
    page_title="Hệ Thống Đối Sánh Lâm Sàng 6 Mạng - Đồ Án AI HUIT",
    page_icon="🩺",
    layout="wide" # Bắt buộc dùng layout rộng để dàn đều 6 cột
)

# =========================================================================
# BẢN VÁ UI: CUSTOM CSS ĐỂ THANH PHẦN TRĂM "ĐẠI" VÀ TRỰC QUAN HƠN 100 LẦN
# =========================================================================
st.markdown("""
    <style>
    /* 1. Tăng độ cao và làm mượt các thanh Progress Bar của Streamlit */
    div[data-testid="ststProgressBar"] > div > div {
        height: 16px !important; /* Tăng độ dày của thanh lên gấp đôi */
        border-radius: 8px !important;
    }
    
    /* 2. Tạo hiệu ứng màu sắc riêng biệt cho cột Mạng Lai Đề Xuất để tạo điểm nhấn */
    div[data-testid="stVerticalBlock"] {
        padding: 10px;
        border-radius: 10px;
    }
    
    /* 3. Phóng to chữ hiển thị các lớp bệnh lý và số phần trăm cho dễ đọc */
    .stCaption {
        font-size: 14px !important;
        font-weight: bold !important;
        color: #1E3A8A !important; /* Đổi sang màu xanh đậm chuyên nghiệp */
    }
    
    /* 4. Tăng khoảng cách dòng giữa các lớp bệnh để không bị dính cục */
    div.element-container {
        margin-bottom: 4px !important;
    }
    </style>
""", unsafe_allow_html=True)

# Danh sách 8 lớp bệnh lý đồng bộ với dữ liệu ISIC
CLASS_NAMES = ['MEL', 'NV', 'BCC', 'AK', 'BKL', 'DF', 'VASC', 'SCC']
CLASS_DESCRIPTIONS = {
    'MEL': "Melanoma (Ung thư hắc tố - Nguy hiểm tính mạng, cần can thiệp gấp)",
    'NV': "Melanocytic Nevi (Nốt ruồi biểu mô lành tính)",
    'BCC': "Basal Cell Carcinoma (Ung thư biểu mô tế bào đáy)",
    'AK': "Actinic Keratosis (Dày sừng quang hóa - Tiền ung thư)",
    'BKL': "Benign Keratosis (Dày sừng lành tính)",
    'DF': "Dermatofibrosis (U xơ da lành tính)",
    'VASC': "Vascular Lesions (Tổn thương mạch máu lành tính)",
    'SCC': "Squamous Cell Carcinoma (Ung thư biểu mô tế bào vảy)"
}

# 📁 Đường dẫn đến trọn bộ 6 file trọng số trong máy HUYPHAT_PC
WEIGHTS_PATHS = {
    'Baseline_CNN_Custom': r'C:\Users\HUYPHAT_PC\Documents\AI\SkinCancer-DeepLearning-ISIC\models\best_baseline.pth',
    'Pure_EfficientNet': r'C:\Users\HUYPHAT_PC\Documents\AI\SkinCancer-DeepLearning-ISIC\models\best_pure_efficientnet.pth',
    'Pure_ViT': r'C:\Users\HUYPHAT_PC\Documents\AI\SkinCancer-DeepLearning-ISIC\models\best_pure_vit.pth',
    'Pure_DeiT': r'C:\Users\HUYPHAT_PC\Documents\AI\SkinCancer-DeepLearning-ISIC\models\best_pure_deit.pth',
    'Hybrid_Model_Proposed': r'C:\Users\HUYPHAT_PC\Documents\AI\SkinCancer-DeepLearning-ISIC\models\best_hybrid_model.pth'
}

# =========================================================================
# 1. HÀM KHỞI TẠO ĐỒNG BỘ CẤU TRÚC LỚP PHÂN LOẠI (8 CLASSES)
# =========================================================================
def create_model_by_name(model_name):
    if model_name == 'Baseline_CNN_Custom':
        # Đồng bộ hóa chính xác cấu trúc mạng tự chế gốc của Phát
        class BaselineCNN(nn.Module):
            def __init__(self):
                super().__init__()
                self.features = nn.Sequential(
                    nn.Conv2d(3, 32, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
                    nn.Conv2d(32, 64, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
                    nn.Conv2d(64, 128, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2)
                )
                self.classifier = nn.Sequential(
                    nn.Flatten(),
                    nn.Linear(128 * 28 * 28, 512),       # Khớp chuẩn kích thước [512, 100352]
                    nn.ReLU(),
                    nn.Dropout(p=0.5),                    # Lớp ẩn trung gian giải quyết việc lệch key index 3 và 4
                    nn.Linear(512, 8)                    # Tầng phân loại đầu ra 8 lớp bệnh
                )
            def forward(self, x):
                return self.classifier(self.features(x))
        model = BaselineCNN()
        
    elif model_name == 'Pure_ResNet50':
        model = models.resnet50(weights=None)
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, 8)
        
    elif model_name == 'Pure_EfficientNet':
        model = models.efficientnet_b0(weights=None)
        in_features = model.classifier[1].in_features
        model.classifier = nn.Sequential(nn.Dropout(p=0.3), nn.Linear(in_features, 8))
        
    elif model_name == 'Pure_ViT':
        model = timm.create_model('vit_tiny_patch16_224', pretrained=False, num_classes=8)
        
    elif model_name == 'Pure_DeiT':
        model = timm.create_model('deit_tiny_distilled_patch16_224', pretrained=False, num_classes=8)
        
    elif model_name == 'Hybrid_Model_Proposed':
        model = HybridSkinModel(num_classes=8)
        
    return model

# =========================================================================
# 2. CACHE TÀI NGUYÊN: NẠP ĐỒNG LOẠT 6 MÔ HÌNH LÊN CARD ĐỒ HỌA
# =========================================================================
@st.cache_resource
def load_all_6_models():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loaded_models = {}
    
    for name, path in WEIGHTS_PATHS.items():
        if os.path.exists(path):
            try:
                model = create_model_by_name(name)
                model.load_state_dict(torch.load(path, map_location=device))
                model = model.to(device)
                model.eval()
                loaded_models[name] = model
            except Exception as e:
                st.sidebar.error(f"❌ Lỗi cấu trúc {name}: {str(e)}")
        else:
            st.sidebar.warning(f"⚠️ Trọng số chưa có: {name}")
            
    return loaded_models, device

# Kích hoạt tiến trình nạp
with st.spinner("📦 Đang nạp đồng loạt 6 kiến trúc mạng lên VRAM hệ thống..."):
    models_dict, device = load_all_6_models()

# =========================================================================
# 3. GIAO DIỆN HIỂN THỊ FRONTEND
# =========================================================================
st.title("🩺 Hệ thống AI chuẩn đoán Ung thư da bằng mô hình (CNN + ViT - CBAM)")
st.markdown("---")

# Sidebar báo cáo hệ thống phần cứng cho HUIT Hội đồng
st.sidebar.header("🖥️ Tài nguyên Máy chủ")
st.sidebar.success(f"Thiết bị: {device.type.upper()}")
st.sidebar.markdown("---")
st.sidebar.header("🤖 Danh sách mạng sẵn sàng:")
for name in WEIGHTS_PATHS.keys():
    if name in models_dict:
        st.sidebar.write(f"✅ {name}")
    else:
        st.sidebar.write(f"❌ {name} (Thiếu file)")

st.write("#### 📥 Tải lên mẫu ảnh da liễu kiểm tra (Out-of-Distribution Sample)")
uploaded_file = st.file_uploader("Kéo thả tệp ảnh nội soi da liễu vào đây", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    st.markdown("---")
    
    # Chia bố cục: Trái hiển thị ảnh gốc - Phải hiển thị bảng tổng quan 6 mạng
    main_col1, main_col2 = st.columns([1, 2])
    
    with main_col1:
        st.write("##### 🖼️ Hình ảnh soi da gốc:")
        image_preview = Image.open(uploaded_file)
        st.image(image_preview, caption=f"File: {uploaded_file.name}", use_column_width=True)
        
    with main_col2:
        st.write("##### 📊 Bảng tổng hợp chẩn đoán đồng thời từ 6 kiến trúc mạng:")
        
        # Tiền xử lý ảnh đồng bộ hoàn toàn
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
        image_cv = Image.open(uploaded_file).convert('RGB')
        input_tensor = transform(image_cv).unsqueeze(0).to(device)
        
        comparison_rows = []
        
        with st.spinner("⚡ 6 Mô hình đang song song tính toán đặc trưng không gian..."):
            for name, model in models_dict.items():
                start_time = time.time()
                with torch.no_grad():
                    with torch.amp.autocast('cuda' if torch.cuda.is_available() else 'cpu'):
                        outputs = model(input_tensor)
                        probs = torch.softmax(outputs, dim=1).cpu().numpy()[0]
                inf_time = (time.time() - start_time) * 1000
                
                best_idx = np.argmax(probs)
                predicted_class = CLASS_NAMES[best_idx]
                confidence_score = probs[best_idx] * 100
                
                comparison_rows.append({
                    "Kiến trúc Mô hình": name,
                    "Chẩn đoán (Top-1)": f"👉 {predicted_class}",
                    "Độ tin cậy (%)": f"{confidence_score:.2f}%",
                    "Độ trễ (Inference)": f"{inf_time:.1f} ms",
                    "Prob_Raw": probs
                })
        
        df_comparison = pd.DataFrame(comparison_rows).drop(columns=["Prob_Raw"])
        st.dataframe(df_comparison, use_container_width=True, hide_index=True)

    # =========================================================================
    # 4. CHI TIẾT THANH PROGRESS BAR 6 CỘT SONG SONG
    # =========================================================================
    st.markdown("---")
    st.write("##### 📈 Chi tiết bản đồ phân phối xác suất trên toàn hệ thống (6-Mô hình):")
    
    # Tạo layout 6 cột đều nhau trên trình duyệt
    columns_list = st.columns(len(comparison_rows))
    
    for idx, row in enumerate(comparison_rows):
        with columns_list[idx]:
            # Đánh dấu highlight rực rỡ riêng cho mạng Lai đề xuất chính của đề tài Phát
            if "Hybrid_Model_Proposed" in row["Kiến trúc Mô hình"]:
                st.markdown(f"🏆 **{row['Kiến trúc Mô hình']}**")
            else:
                st.markdown(f"🔹 **{row['Kiến trúc Mô hình']}**")
                
            st.metric(label="Dự đoán", value=row["Chẩn đoán (Top-1)"], delta=row["Độ tin cậy (%)"])
            st.caption(f"Độ trễ: {row['Độ trễ (Inference)']}")
            
            # Xuất thanh tiến trình xác suất của 8 lớp bệnh dưới cột mô hình tương ứng
            # model_probs = row["Prob_Raw"]
            # for c_idx, c_name in enumerate(CLASS_NAMES):
            #     val_p = float(model_probs[c_idx])
            #     st.write(f"**{c_name}**: {val_p*100:.1f}%")
            #     st.progress(val_p)

            # 🔄 THAY THẾ BẰNG ĐOẠN HTML/CSS DIỄN HOẠ SIÊU TRỰC QUAN NÀY:
            model_probs = row["Prob_Raw"]
            for c_idx, c_name in enumerate(CLASS_NAMES):
                val_p = float(model_probs[c_idx])
                percentage = val_p * 100
                
                # 🎯 LẤY TÊN ĐẦY ĐỦ CỦA BỆNH TỪ DICTIONARY (Nếu không tìm thấy sẽ dùng tạm tên viết tắt)
                full_disease_name = CLASS_DESCRIPTIONS.get(c_name, c_name)
                
                # Tự động tính toán màu sắc dựa trên độ nguy hiểm và độ tin cậy
                if percentage > 50:
                    bar_color = "#10B981" # Xanh lá đậm nổi bật khi đoán mạnh
                    if c_name in ['MEL', 'BCC', 'SCC']:
                        bar_color = "#EF4444" # Đỏ rực cảnh báo nếu đoán ra Ung thư nguy hiểm
                else:
                    bar_color = "#3B82F6" # Màu xanh dương tiêu chuẩn cho các lớp phụ
                
                # Render thanh tiến trình "Đại" tùy biến bằng HTML sạch
                # Thêm thuộc tính title để khi di chuột vào vẫn thấy rõ ràng, tối ưu giao diện hẹp
                st.markdown(f"""
                    <div style="margin-bottom: 12px;" title="{full_disease_name}">
                        <div style="display: flex; justify-content: space-between; font-size: 12px; font-weight: bold; margin-bottom: 2px; line-height: 1.3;">
                            <span style="color: #1E3A8A; padding-right: 5px; word-break: break-word;">{full_disease_name}</span>
                            <span style="color: {bar_color}; white-space: nowrap;">{percentage:.1f}%</span>
                        </div>
                        <div style="background-color: #E5E7EB; width: 100%; height: 14px; border-radius: 7px; overflow: hidden;">
                            <div style="background-color: {bar_color}; width: {percentage}%; height: 100%; border-radius: 7px; transition: width 0.5s ease-in-out;"></div>
                        </div>
                    </div>
                """, unsafe_allow_html=True)