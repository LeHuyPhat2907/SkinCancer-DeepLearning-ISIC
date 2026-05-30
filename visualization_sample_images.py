import pandas as pd
import matplotlib.pyplot as plt
import cv2
import os

# 1. Load metadata
df = pd.read_csv(r'd:\SkinCancer_AI_ISIC\data\metadata_refined.csv')
img_dir = r'd:\SkinCancer_AI_ISIC\data_processed\final_cleaned'

# 2. Lấy danh sách 8 lớp
classes = df['diagnosis_code'].unique()

# 3. Vẽ Grid ảnh 2x4
plt.figure(figsize=(20, 10))
plt.suptitle("Bảng so sánh mẫu ảnh 08 lớp bệnh lý (Sau Tiền xử lý)", fontsize=20, fontweight='bold')

for i, cls in enumerate(classes):
    # Lấy ngẫu nhiên 1 tấm của lớp này
    sample = df[df['diagnosis_code'] == cls].sample(1).iloc[0]
    img_id = sample['isic_id']
    
    img_path = os.path.join(img_dir, f"{img_id}.jpg")
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    plt.subplot(2, 4, i+1)
    plt.imshow(img)
    plt.title(f"Class: {cls}\nID: {img_id}", fontsize=14, color='darkred')
    plt.axis('off')

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig(r'd:\SkinCancer_AI_ISIC\data\sample_images_grid.png')
plt.show()