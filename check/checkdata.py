import os
import pandas as pd

# Đường dẫn đến thư mục data của bạn
data_dir = 'data' 
metadata_path = 'data/metadata.csv' # Thay đổi tên file đúng với file bạn tải về

def final_check():
    # 1. Đếm số lượng ảnh thực tế
    valid_extensions = ('.jpg', '.jpeg', '.png')
    all_files = [f for root, dirs, files in os.walk(data_dir) for f in files if f.lower().endswith(valid_extensions)]
    print(f"✅ Tổng số lượng ảnh đã tải: {len(all_files):,}")

    # 2. Kiểm tra Metadata
    if os.path.exists(metadata_path):
        df = pd.read_csv(metadata_path)
        print(f"✅ Số dòng trong Metadata: {len(df):,}")
        print("\n📊 Thống kê số lượng theo từng lớp bệnh:")
        # Thay 'dx' bằng tên cột nhãn trong file csv của bạn
        if 'dx' in df.columns:
            print(df['dx'].value_counts())
    else:
        print("⚠️ Không tìm thấy file metadata.csv trong thư mục!")

final_check()