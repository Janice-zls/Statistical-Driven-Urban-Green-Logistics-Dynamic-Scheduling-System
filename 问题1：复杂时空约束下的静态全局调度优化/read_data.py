import pandas as pd
import os

base_path = r"g:\B.比赛\2026第十八届“华中杯”大学生数学建模挑战赛\A题：城市绿色物流配送调度_1776844160973\附件"
files = ["订单信息.xlsx", "距离矩阵.xlsx", "客户坐标信息.xlsx", "时间窗.xlsx"]

for f in files:
    print(f"--- {f} ---")
    df = pd.read_excel(os.path.join(base_path, f))
    print(df.head())
    print("\n")
