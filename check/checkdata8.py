import pandas as pd
df = pd.read_csv(r'd:\SkinCancer_AI_ISIC\data\metadata_refined.csv')
print("🔍 Kiểm tra lớp AK:")
print(df[df['diagnosis_code'] == 'AK'].shape[0])