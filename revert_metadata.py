import pandas as pd
import json

path_meta_goc = r'd:\SkinCancer_AI_ISIC\data\metadata.csv'
path_meta_refined = r'd:\SkinCancer_AI_ISIC\data\metadata_refined.csv'

# 1. Đọc dữ liệu
df_goc = pd.read_csv(path_meta_goc, low_memory=False)
df_refined = pd.read_csv(path_meta_refined)

# 2. Bảng ánh xạ gom nhóm về 9 mã chuẩn
diagnosis_to_9class = {
    'Melanoma, NOS': 'MEL', 'Melanoma in situ': 'MEL', 'Melanoma Invasive': 'MEL',
    'Nevus': 'NV',
    'Basal cell carcinoma': 'BCC',
    'Solar or actinic keratosis': 'AK',
    'Seborrheic keratosis': 'BKL', 'Pigmented benign keratosis': 'BKL', 'Solar lentigo': 'BKL',
    'Dermatofibroma': 'DF',
    'Vascular lesion': 'VASC',
    'Squamous cell carcinoma, NOS': 'SCC'
}

# 3. Tạo nhãn trên file GỐC trước
df_goc['new_diag_code'] = df_goc['diagnosis_3'].map(diagnosis_to_9class).fillna('UNK')

# Vét thêm từ diagnosis_2 cho DF và VASC
df_goc.loc[(df_goc['new_diag_code'] == 'UNK') & (df_goc['diagnosis_2'].str.contains('Fibro-histiocytic', na=False)), 'new_diag_code'] = 'DF'
df_goc.loc[(df_goc['new_diag_code'] == 'UNK') & (df_goc['diagnosis_2'].str.contains('Vascular', na=False)), 'new_diag_code'] = 'VASC'

# 4. DỌN DẸP file refined: Xóa các cột cũ nếu đã lỡ tạo ở các lần chạy trước
cols_to_remove = ['diagnosis_code', 'new_diag_code', 'label', 'diagnosis_1', 'diagnosis_3']
df_refined = df_refined.drop(columns=[c for c in cols_to_remove if c in df_refined.columns])

# 5. GỘP (Merge)
# Chỉ lấy isic_id và cột nhãn mới từ file gốc
mapping_subset = df_goc[['isic_id', 'new_diag_code']]
df_final = pd.merge(df_refined, mapping_subset, on='isic_id', how='left')

# Đổi tên lại cho đẹp
df_final = df_final.rename(columns={'new_diag_code': 'diagnosis_code'})

# 6. Tạo nhãn số (0-8)
label_mapping = {
    "MEL": 0, "NV": 1, "BCC": 2, "AK": 3, 
    "BKL": 4, "DF": 5, "VASC": 6, "SCC": 7, "UNK": 8
}
df_final['label'] = df_final['diagnosis_code'].map(label_mapping).fillna(8).astype(int)

# 7. Lưu Mapping và Metadata
with open(r'd:\SkinCancer_AI_ISIC\data\label_mapping_9class.json', 'w') as f:
    json.dump(label_mapping, f)

df_final.to_csv(path_meta_refined, index=False)

print("✅ ĐÃ KHÔI PHỤC XONG 09 LỚP!")
print("\n📊 Thống kê trên 51,620 ảnh của Phát:")
print(df_final['diagnosis_code'].value_counts())