import os
from PIL import Image
import pandas as pd

# Đường dẫn
image_dir = r'd:\SkinCancer_AI_ISIC\data\raw_images'
log_file = r'd:\SkinCancer_AI_ISIC\corrupted_files_log.csv'

corrupted_files = []
valid_count = 0

print(f"🚀 Đang kiểm tra 51,620 ảnh...")

for filename in os.listdir(image_dir):
    if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
        file_path = os.path.join(image_dir, filename)
        try:
            with Image.open(file_path) as img:
                img.verify()  # Kiểm tra file có bị hỏng không
            valid_count += 1
        except (IOError, SyntaxError) as e:
            print(f"❌ Phát hiện file lỗi: {filename}")
            corrupted_files.append({"file_name": filename, "error": str(e)})

# Ghi log file lỗi ra CSV như Note yêu cầu
if corrupted_files:
    df_err = pd.DataFrame(corrupted_files)
    df_err.to_csv(log_file, index=False)
    print(f"⚠️ Đã log {len(corrupted_files)} file lỗi vào {log_file}")
else:
    print("✅ Tuyệt vời! Không phát hiện file ảnh nào bị lỗi.")

print(f"📊 Số lượng ảnh hợp lệ còn lại: {valid_count}")