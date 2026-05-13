import pandas as pd
from sklearn.model_selection import train_test_split

# 1. Load metadata chuẩn 08 lớp
df = pd.read_csv(r'd:\SkinCancer_AI_ISIC\data\metadata_refined.csv')

# 2. Chia dữ liệu theo tỷ lệ 70/20/10
# Bước 1: Tách 70% cho Train, 30% còn lại cho (Val + Test)
train_df, temp_df = train_test_split(
    df, 
    test_size=0.30, 
    random_state=42, 
    stratify=df['label'] # Giữ nguyên tỉ lệ 08 lớp
)

# Bước 2: Chia 30% temp_df thành Val (20%) và Test (10%)
# (Lấy 1/3 của 30% sẽ ra 10% cho Test)
val_df, test_df = train_test_split(
    temp_df, 
    test_size=1/3, 
    random_state=42, 
    stratify=temp_df['label']
)

# 3. Save split files (Sub-task 3)
train_df.to_csv(r'd:\SkinCancer_AI_ISIC\data\train.csv', index=False)
val_df.to_csv(r'd:\SkinCancer_AI_ISIC\data\val.csv', index=False)
test_df.to_csv(r'd:\SkinCancer_AI_ISIC\data\test.csv', index=False)

print("✅ Đã chia và lưu file split thành công!")
print(f"📊 Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")

print("\n🔥 Tỉ lệ lớp trong tập Test (nên giống tập Train):")
print(test_df['diagnosis_code'].value_counts(normalize=True) * 100)