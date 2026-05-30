import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Load metadata gốc (vì file gốc chứa đầy đủ thông tin age, sex hơn file tinh gọn)
df_goc = pd.read_csv(r'd:\SkinCancer_AI_ISIC\data\metadata.csv', low_memory=False)

# Thiết lập layout cho 3 biểu đồ
fig, axes = plt.subplots(1, 3, figsize=(20, 6))

# Sub-task 1: Analyze age distribution
sns.histplot(df_goc['age_approx'].dropna(), bins=20, kde=True, ax=axes[0], color='skyblue')
axes[0].set_title('Phân bổ độ tuổi bệnh nhân', fontsize=14)
axes[0].set_xlabel('Tuổi (Xấp xỉ)')
axes[0].set_ylabel('Số lượng')

# Sub-task 2: Analyze gender distribution
df_goc['sex'].value_counts().plot.pie(autopct='%1.1f%%', ax=axes[1], colors=['lightcoral', 'cornflowerblue'], startangle=90)
axes[1].set_title('Phân bổ giới tính', fontsize=14)
axes[1].set_ylabel('')

# Sub-task 3: Analyze lesion localization
sns.countplot(data=df_goc, y='anatom_site_general', order=df_goc['anatom_site_general'].value_counts().index, ax=axes[2], palette='viridis')
axes[2].set_title('Vị trí tổn thương trên cơ thể', fontsize=14)
axes[2].set_xlabel('Số lượng')
axes[2].set_ylabel('Vị trí')

plt.tight_layout()
plt.savefig(r'd:\SkinCancer_AI_ISIC\data\metadata_eda.png')
plt.show()