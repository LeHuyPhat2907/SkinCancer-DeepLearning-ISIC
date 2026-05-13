import pandas as pd

metadata_path = r'd:\SkinCancer_AI_ISIC\data\metadata.csv'
cols = ['isic_id', 'diagnosis_1', 'image_type']

# Đọc dữ liệu
df = pd.read_csv(metadata_path, usecols=cols, low_memory=False)

# 1. Chỉ lấy ảnh Dermoscopic (Chuẩn y khoa cho AI)
df_refined = df[df['image_type'] == 'dermoscopic'].copy()

# 2. Loại bỏ các dòng không rõ ràng (Indeterminate)
df_refined = df_refined[df_refined['diagnosis_1'] != 'Indeterminate']

# 3. Cân bằng dữ liệu (Lấy toàn bộ Malignant, và khoảng 30k Benign ngẫu nhiên)
malignant_df = df_refined[df_refined['diagnosis_1'] == 'Malignant']
benign_df = df_refined[df_refined['diagnosis_1'] == 'Benign'].sample(n=min(30000, len(df_refined)), random_state=42)

df_final = pd.concat([malignant_df, benign_df])

# Lưu file metadata mới
df_final.to_csv(r'd:\SkinCancer_AI_ISIC\data\metadata_refined.csv', index=False)

print(f"✅ Đã tạo xong metadata_refined.csv với {len(df_final):,} ảnh.")
print(df_final['diagnosis_1'].value_counts())