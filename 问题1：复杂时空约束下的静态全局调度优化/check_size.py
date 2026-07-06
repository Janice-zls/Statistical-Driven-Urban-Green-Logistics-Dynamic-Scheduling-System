from PIL import Image
import os

img1_path = r"g:\B.比赛\2026第十八届华中杯大学生数学建模挑战赛\A题城市绿色物流配送调度_1776844160973\问题1\图1_城市配送时空网络拓扑与需求特征.png"
img2_path = r"g:\B.比赛\2026第十八届华中杯大学生数学建模挑战赛\A题城市绿色物流配送调度_1776844160973\问题1\图1_复合视窗_城市调度空间与容量联合分析面板.png"

img1 = Image.open(img1_path)
img2 = Image.open(img2_path)

print(f"Image 1: {img1.size}")
print(f"Image 2: {img2.size}")
