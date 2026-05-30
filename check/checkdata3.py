import pandas as pd

metadata_path = r'd:\SkinCancer_AI_ISIC\data\metadata.csv'

# Chỉ đọc các cột cần thiết để tiết kiệm RAM
cols_to_use = ['isic_id', 'diagnosis_1', 'image_type']

df = pd.read_csv(metadata_path, usecols=cols_to_use, low_memory=False)

print("📊 THỐNG KÊ SỐ LƯỢNG THEO TỪNG LOẠI BỆNH (DIAGNOSIS_1):")
print(df['diagnosis_1'].value_counts())

print("\n📸 THỐNG KÊ THEO LOẠI ẢNH (IMAGE_TYPE):")
print(df['image_type'].value_counts())