import pandas as pd
path_meta_goc = r'd:\SkinCancer_AI_ISIC\data\metadata.csv'
df_test = pd.read_csv(path_meta_goc, nrows=5) # Chỉ đọc 5 dòng cho nhẹ
print(df_test.columns.tolist())