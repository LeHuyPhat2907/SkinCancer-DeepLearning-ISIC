import pandas as pd

path_meta_goc = r'd:\SkinCancer_AI_ISIC\data\metadata.csv'
df_goc = pd.read_csv(path_meta_goc, low_memory=False)

# Kiểm tra thử các cột diagnosis khác
cols_to_check = ['diagnosis_1', 'diagnosis_2', 'diagnosis_3', 'diagnosis_4', 'diagnosis_5']

print("🔍 Kiểm tra giá trị ở các cột diagnosis phụ:")
for col in cols_to_check:
    if col in df_goc.columns:
        print(f"\n--- Cột {col} ---")
        print(df_goc[col].value_counts().head(10))

# Kiểm tra xem có cột nào chứa tên các loại bệnh (MEL, BCC...) không
keywords = ['Melanoma', 'Nevus', 'Carcinoma', 'Keratosis', 'BCC', 'MEL', 'NV']
print("\n🔍 Đang quét tìm các từ khóa bệnh lý trong toàn bộ file...")
for col in df_goc.columns:
    sample_val = str(df_goc[col].iloc[0]).lower()
    if any(k.lower() in sample_val for k in keywords):
        print(f"-> Tìm thấy manh mối ở cột: [{col}]")