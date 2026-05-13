import os
import cv2
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Quét danh sách ảnh (Bạn nên quét trên folder chứa ảnh gốc GỐC chưa resize)
path_raw = r'd:\SkinCancer_AI_ISIC\data\raw_images' # Điều chỉnh đường dẫn folder gốc của Phát
image_files = [f for f in os.listdir(path_raw) if f.endswith('.jpg')][:2000] # Quét mẫu 2000 ảnh để lấy đại diện

widths = []
heights = []

print("🔍 Đang quét kích thước ảnh mẫu...")
for img_name in image_files:
    img = cv2.imread(os.path.join(path_raw, img_name))
    if img is not None:
        h, w, _ = img.shape
        widths.append(w)
        heights.append(h)

# 2. Thống kê vào DataFrame
df_sizes = pd.DataFrame({'Width': widths, 'Height': heights})

# 3. Visualization size distribution
plt.figure(figsize=(12, 5))

# Biểu đồ phân bổ Width/Height
plt.subplot(1, 2, 1)
sns.kdeplot(data=df_sizes, x='Width', y='Height', fill=True, cmap='Blues')
plt.title('Mật độ phân bổ kích thước ảnh gốc', fontsize=12)

# Biểu đồ Scatter để thấy sự đa dạng
plt.subplot(1, 2, 2)
plt.scatter(df_sizes['Width'], df_sizes['Height'], alpha=0.3, s=10, c='orange')
plt.axvline(224, color='red', linestyle='--', label='Target size (224)')
plt.axhline(224, color='red', linestyle='--')
plt.title('Biểu đồ phân tán Width vs Height', fontsize=12)
plt.legend()

plt.tight_layout()
plt.savefig(r'd:\SkinCancer_AI_ISIC\data\image_size_distribution.png')
plt.show()

print(f"📊 Kích thước trung bình: {int(df_sizes['Width'].mean())}x{int(df_sizes['Height'].mean())}")
print(f"📏 Kích thước lớn nhất: {df_sizes['Width'].max()}x{df_sizes['Height'].max()}")