import pandas as pd
import json

# Đường dẫn file
metadata_path = r'd:\SkinCancer_AI_ISIC\data\metadata_refined.csv'
mapping_save_path = r'd:\SkinCancer_AI_ISIC\data\label_mapping.json'

# 1. Đọc metadata
df = pd.read_csv(metadata_path)

# 2. Tạo dictionary ánh xạ (Mapping)
# Chúng ta chọn Benign là 0 (lớp âm tính) và Malignant là 1 (lớp dương tính)
label_mapping = {
    "Benign": 0,
    "Malignant": 1
}

# 3. Áp dụng Encoding (Mapping class name -> integer)
df['label'] = df['diagnosis_1'].map(label_mapping)

# 4. Kiểm tra xem có dòng nào bị lỗi không (null sau khi map)
if df['label'].isnull().any():
    print("⚠️ Cảnh báo: Có dòng không khớp nhãn, hãy kiểm tra lại!")
else:
    # 5. Lưu lại Metadata mới và file Mapping
    df.to_csv(metadata_path, index=False)
    
    with open(mapping_save_path, 'w') as f:
        json.dump(label_mapping, f)
        
    print("✅ Đã Encode nhãn thành công!")
    print(f"📊 Bảng ánh xạ: {label_mapping}")
    print(df[['diagnosis_1', 'label']].head())