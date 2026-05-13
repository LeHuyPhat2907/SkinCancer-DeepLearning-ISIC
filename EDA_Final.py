import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Load dữ liệu
df = pd.read_csv(r'd:\SkinCancer_AI_ISIC\data\metadata_refined.csv')
counts = df['diagnosis_code'].value_counts()
total = len(df)
percentages = (counts / total * 100).round(2)

# 2. Tạo DataFrame thống kê
df_stats = pd.DataFrame({
    'Class': counts.index,
    'Count': counts.values,
    'Percentage': percentages.values
})

# 3. Visualization (Bar chart class counts)
plt.figure(figsize=(14, 8))
colors = ['#2c3e50' if x > 1000 else '#e74c3c' for x in df_stats['Count']] # Đỏ cho lớp thiểu số
ax = sns.barplot(data=df_stats, x='Class', y='Count', palette=colors)

# Thêm số lượng và % trên đầu cột (So sánh tỷ lệ classes)
for i, p in enumerate(ax.patches):
    ax.annotate(f'{int(p.get_height())}\n({percentages[i]}%)', 
                (p.get_x() + p.get_width() / 2., p.get_height()), 
                ha = 'center', va = 'center', 
                xytext = (0, 15), 
                textcoords = 'offset points',
                fontsize=11, fontweight='bold')

plt.title('Thống kê Class Imbalance trong tập dữ liệu 51,620 ảnh', fontsize=16, fontweight='bold', pad=20)
plt.xlabel('Mã loại bệnh', fontsize=12)
plt.ylabel('Số lượng mẫu', fontsize=12)
plt.ylim(0, max(counts) * 1.15) # Tạo khoảng trống phía trên để ghi text

# Chú thích minority
plt.text(6.5, 5000, '🔴: Minority Classes\n(Cần xử lý Weighted Loss)', 
         bbox=dict(facecolor='white', alpha=0.5), color='#e74c3c', fontweight='bold')

plt.tight_layout()
plt.savefig(r'd:\SkinCancer_AI_ISIC\data\class_imbalance_eda.png')
plt.show()