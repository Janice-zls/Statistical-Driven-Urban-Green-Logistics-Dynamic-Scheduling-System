import pandas as pd
import os

files = ["订单信息.xlsx", "距离矩阵.xlsx", "客户坐标信息.xlsx", "时间窗.xlsx"]

for f in files:
    print(f"--- {f} ---")
    try:
        df = pd.read_excel(f)
        print(df.head())
        print(df.columns.tolist())
        print("\n")
    except Exception as e:
        print(e)
