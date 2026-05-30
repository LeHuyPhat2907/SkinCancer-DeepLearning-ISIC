import os
import pandas as pd

# Đường dẫn
metadata_path = r'd:\SkinCancer_AI_ISIC\data\metadata_refined.csv'
image_dir = r'd:\SkinCancer_AI_ISIC\data\raw_images'

# Đọc metadata
df = pd.read_csv(metadata_path)
metadata_ids = set(df['isic_id'].astype(str).tolist())

# Đọc file thực tế (bỏ đuôi .jpg)
folder_files = [os.path.splitext(f)[0] for f in os.listdir(image_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
folder_ids = set(folder_files)

print(f"📊 Kiểm tra Mapping cho {len(metadata_ids)} dòng Metadata và {len(folder_ids)} file ảnh...")

# 1. Check Metadata có nhưng Folder không có
missing_in_folder = metadata_ids - folder_ids
# 2. Check Folder có nhưng Metadata không có
extra_in_folder = folder_ids - metadata_ids

if not missing_in_folder and not extra_in_folder:
    print("✅ HOÀN HẢO! Mapping 1-1 khớp hoàn toàn.")
else:
    if missing_in_folder:
        print(f"⚠️ Metadata thừa {len(missing_in_folder)} dòng (không có ảnh thực tế).")
        # Remove unmatched samples khỏi Metadata
        df = df[~df['isic_id'].isin(missing_in_folder)]
        df.to_csv(metadata_path, index=False)
        print("   -> Đã xóa các dòng thừa trong Metadata.")
        
    if extra_in_folder:
        print(f"⚠️ Folder thừa {len(extra_in_folder)} file ảnh (không có trong Metadata).")
        # Xóa file thừa để folder sạch sẽ
        for extra_id in extra_in_folder:
            for ext in ['.jpg', '.jpeg', '.png']:
                file_to_del = os.path.join(image_dir, extra_id + ext)
                if os.path.exists(file_to_del):
                    os.remove(file_to_del)
        print("   -> Đã xóa các file ảnh thừa trong Folder.")

print(f"🚀 Kết quả cuối cùng: {len(df)} mẫu dữ liệu đã sẵn sàng 1-1!")