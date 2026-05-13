import os
import pandas as pd

# Đường dẫn mới sau khi dọn dẹp
refined_metadata_path = r'd:\SkinCancer_AI_ISIC\data_refined\metadata_refined.csv'
refined_images_dir = r'd:\SkinCancer_AI_ISIC\data_refined'

def final_validation():
    # 1. Đọc metadata đã lọc
    df = pd.read_csv(refined_metadata_path)
    expected_count = len(df)
    
    # 2. Đếm số lượng file thực tế trong thư mục mới
    valid_extensions = ('.jpg', '.jpeg', '.png')
    actual_files = [f for f in os.listdir(refined_images_dir) if f.lower().endswith(valid_extensions)]
    actual_count = len(actual_files)
    
    print("--- BÁO CÁO NGHIỆM THU TASK 15 ---")
    print(f"📍 Số lượng ảnh theo Metadata: {expected_count:,}")
    print(f"📍 Số lượng ảnh thực tế trong folder: {actual_count:,}")
    
    if expected_count == actual_count:
        print("✅ KHỚP DỮ LIỆU! Bạn đã sẵn sàng sang Task 16.")
    else:
        print(f"⚠️ LỆCH DỮ LIỆU! Thiếu {expected_count - actual_count} ảnh.")

    print("\n📊 PHÂN BỔ NHÃN CUỐI CÙNG:")
    print(df['diagnosis_1'].value_counts())
    
    # Tính tỷ lệ % để đưa vào báo cáo
    print("\n📈 TỶ LỆ PHẦN TRĂM:")
    print(df['diagnosis_1'].value_counts(normalize=True) * 100)

final_validation()