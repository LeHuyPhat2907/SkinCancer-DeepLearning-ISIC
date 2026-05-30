import pandas as pd

# Đường dẫn đến file metadata của bạn
metadata_path = r'd:\SkinCancer_AI_ISIC\data\metadata.csv' 

try:
    # Chỉ đọc 5 dòng đầu để lấy tên cột (Không lo bị lag máy)
    df_sample = pd.read_csv(metadata_path, nrows=5, low_memory=False)
    
    print("--- 5 DÒNG ĐẦU TIÊN ---")
    print(df_sample.head())
    
    print("\n--- DANH SÁCH TÊN CỘT CHÍNH XÁC ---")
    print(df_sample.columns.tolist())

except Exception as e:
    print(f"Lỗi rồi: {e}")