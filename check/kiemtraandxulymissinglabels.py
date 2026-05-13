import pandas as pd

# Đường dẫn file
metadata_path = r'd:\SkinCancer_AI_ISIC\data\metadata_refined.csv'

# Đọc metadata
df = pd.read_csv(metadata_path)

print(f"🚀 Bắt đầu quét nhãn cho {len(df)} dòng dữ liệu...")

# 1. Phát hiện giá trị Null (Scan label columns & Detect null values)
null_count = df['diagnosis_1'].isnull().sum()

if null_count > 0:
    print(f"⚠️ Cảnh báo: Phát hiện {null_count} dòng bị thiếu nhãn (null).")
    
    # 2. Xử lý thiếu nhãn (Handle missing labels)
    # Vì đây là dữ liệu y khoa, ta không nên tự ý điền nhãn mà nên loại bỏ để đảm bảo độ chính xác
    df_cleaned = df.dropna(subset=['diagnosis_1'])
    
    # Lưu lại file sạch hoàn toàn
    df_cleaned.to_csv(metadata_path, index=False)
    print(f"✅ Đã loại bỏ các dòng thiếu nhãn. Số lượng còn lại: {len(df_cleaned)}")
else:
    print("✅ Tuyệt vời! Không có dòng nào bị thiếu nhãn (Missing labels: 0).")

# Kiểm tra thêm nhãn rỗng (trường hợp không phải null nhưng để trống)
empty_string_count = (df['diagnosis_1'] == '').sum()
print(f"📍 Số lượng nhãn rỗng (empty string): {empty_string_count}")