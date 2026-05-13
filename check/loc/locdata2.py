import os
import shutil
import pandas as pd

# Đường dẫn
src_dir = r'd:\SkinCancer_AI_ISIC\data' # Nơi chứa 550k ảnh cũ
dst_dir = r'd:\SkinCancer_AI_ISIC\data_refined' # Nơi lưu 50k ảnh "tinh hoa"
metadata_refined_path = r'd:\SkinCancer_AI_ISIC\data\metadata_refined.csv'

# Tạo thư mục đích nếu chưa có
if not os.path.exists(dst_dir):
    os.makedirs(dst_dir)

# Đọc danh sách file cần giữ
df = pd.read_csv(metadata_refined_path)
files_to_keep = set(df['isic_id'].tolist())

print(f"🚀 Đang bắt đầu lọc và di chuyển {len(files_to_keep)} ảnh...")

count = 0
# Duyệt qua các thư mục ảnh cũ
for root, dirs, files in os.walk(src_dir):
    for file in files:
        file_id = os.path.splitext(file)[0]
        if file_id in files_to_keep:
            shutil.move(os.path.join(root, file), os.path.join(dst_dir, file))
            count += 1
            if count % 1000 == 0:
                print(f"Đã di chuyển: {count} ảnh")

print(f"✅ Hoàn tất! Đã đưa {count} ảnh vào thư mục 'data_refined'.")