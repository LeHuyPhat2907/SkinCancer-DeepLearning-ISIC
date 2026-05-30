import pandas as pd

path_meta_goc = r'd:\SkinCancer_AI_ISIC\data\metadata.csv'
df_goc = pd.read_csv(path_meta_goc, low_memory=False)

# In ra 20 giá trị xuất hiện nhiều nhất trong cột diagnosis_1
print("🔍 Các giá trị thực tế trong cột diagnosis_1 là:")
print(df_goc['diagnosis_1'].value_counts().head(20))