from PIL import Image
import numpy as np
import os

def remove_background(img_path, out_path, tolerance=20):
    print(f"Reading image: {img_path}")
    img = Image.open(img_path).convert("RGBA")
    data = np.array(img)
    
    # 假设图片左上角像素(0, 0)为背景色
    bg_color = data[0, 0, :3]
    print(f"Detected background color: {bg_color}")
    
    # 计算所有像素与背景色的差异
    r_diff = np.abs(data[:,:,0].astype(int) - bg_color[0])
    g_diff = np.abs(data[:,:,1].astype(int) - bg_color[1])
    b_diff = np.abs(data[:,:,2].astype(int) - bg_color[2])
    
    # 找到与背景色接近的像素（容差值 tolerance）
    mask = (r_diff <= tolerance) & (g_diff <= tolerance) & (b_diff <= tolerance)
    
    # 将匹配背景色的像素改为纯白色，并且Alpha通道设为0（完全透明）
    data[mask, 0] = 255 # R
    data[mask, 1] = 255 # G
    data[mask, 2] = 255 # B
    data[mask, 3] = 0   # A (0为透明)
    
    out_img = Image.fromarray(data)
    out_img.save(out_path, "PNG")
    print(f"Saved transparent image to: {out_path}")

if __name__ == "__main__":
    base_dir = r"g:\B.比赛\2026第十八届华中杯大学生数学建模挑战赛\A题城市绿色物流配送调度_1776844160973\问题1"
    img_path = os.path.join(base_dir, "最终成果_极简高级组合大屏面板.png")
    out_path = os.path.join(base_dir, "最终成果_极简高级组合大屏面板_白色透明背景.png")
    
    if os.path.exists(img_path):
        remove_background(img_path, out_path)
    else:
        print(f"Error: Could not find {img_path}")
