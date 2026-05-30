import pandas as pd
import json
import numpy as np

# 1. Đọc metadata hiện tại
path_refined = r'd:\SkinCancer_AI_ISIC\data\metadata_refined.csv'
df = pd.read_csv(path_refined)

# 2. Định nghĩa lại Mapping 08 lớp (Bỏ AK)
new_label_mapping = {
    "MEL": 0, "NV": 1, "BCC": 2, "BKL": 3, 
    "DF": 4, "VASC": 5, "SCC": 6, "UNK": 7
}

# 3. Cập nhật nhãn số
df['label'] = df['diagnosis_code'].map(new_label_mapping).astype(int)

# 4. Tính Class Weights để xử lý Imbalance
counts = df['label'].value_counts().sort_index().values
weights = 1. / counts
weights = weights / weights.sum() * len(counts) # Chuẩn hóa về mức trung bình

# 5. Lưu Mapping và Metadata mới
with open(r'd:\SkinCancer_AI_ISIC\data\label_mapping_8class.json', 'w') as f:
    json.dump(new_label_mapping, f)
df.to_csv(path_refined, index=False)

print("✅ ĐÃ CHUYỂN SANG 08 LỚP THÀNH CÔNG!")
print(f"📊 Class Weights cho hàm Loss: \n{weights.tolist()}")