import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# 1. Load dữ liệu
df = pd.read_csv(r'd:\SkinCancer_AI_ISIC\data\metadata_refined.csv')
counts = df['diagnosis_code'].value_counts()
labels = counts.index
values = counts.values

# 2. Giả lập tác động của Class Weights (mô phỏng sự cân bằng)
# Sau khi áp dụng weights, "tầm ảnh hưởng" của các lớp sẽ xấp xỉ bằng nhau
weighted_impact = [1.0] * len(labels) 

# 3. Vẽ biểu đồ
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7))

# Biểu đồ 1: Trước cân bằng
sns.barplot(x=labels, y=values, ax=ax1, palette='magma')
ax1.set_title('Hình A: Phân bổ dữ liệu gốc (Imbalanced)', fontsize=14, fontweight='bold')
ax1.set_ylabel('Số lượng mẫu (Images)')
for i, v in enumerate(values):
    ax1.text(i, v + 100, str(v), ha='center', fontweight='bold')

# Biểu đồ 2: Sau khi áp dụng Weighted Loss
sns.barplot(x=labels, y=weighted_impact, ax=ax2, palette='viridis')
ax2.set_title('Hình B: Tầm ảnh hưởng sau Weighted Loss (Balanced)', fontsize=14, fontweight='bold')
ax2.set_ylabel('Trọng số ảnh hưởng tương đối (Scale)')
ax2.set_ylim(0, 1.5)
for i, v in enumerate(weighted_impact):
    ax2.text(i, v + 0.05, "Balanced", ha='center', fontsize=9)

plt.tight_layout()
plt.savefig(r'd:\SkinCancer_AI_ISIC\data\class_distribution_comparison.png')
plt.show()