import os
import pandas as pd

# Đường dẫn (Phát điều chỉnh cho đúng thực tế trên máy nhé)
path_raw = r'd:\SkinCancer_AI_ISIC\data_raw' # Giả sử đây là folder ảnh gốc
path_cleaned = r'd:\SkinCancer_AI_ISIC\data_processed\final_cleaned'
metadata_path = r'd:\SkinCancer_AI_ISIC\data\metadata_refined.csv'

def get_size_gb(start_path='.'):
    total_size = 0
    for dirpath, dirnames, filenames in os.walk(start_path):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            total_size += os.path.getsize(fp)
    return round(total_size / (1024**3), 2)

# Sub-task 1 & 2: Đếm số lượng và tính dung lượng
num_raw = len(os.listdir(path_raw)) if os.path.exists(path_raw) else 0
num_cleaned = len(os.listdir(path_cleaned))
size_cleaned = get_size_gb(path_cleaned)

# Thống kê nhãn từ metadata
df = pd.read_csv(metadata_path)
stats = df['diagnosis_code'].value_counts()

# 3. Tạo bảng so sánh (Note: Preprocessing comparison table)
print("--- 📊 BẢNG THỐNG KÊ TIỀN XỬ LÝ ---")
data = {
    "Chỉ số": ["Số lượng ảnh", "Dung lượng (GB)", "Số lượng lớp (Classes)", "Kích thước ảnh"],
    "Trước xử lý": [num_raw if num_raw > 0 else "N/A", "23.0 (approx)", "N/A", "Variable"],
    "Sau xử lý": [num_cleaned, size_cleaned, len(stats), "224x224 (Fixed)"]
}
comparison_df = pd.DataFrame(data)
print(comparison_df.to_string(index=False))

print("\n--- 🏷️ CHI TIẾT 08 LỚP BỆNH LÝ (FINAL CLEANED) ---")
print(stats)