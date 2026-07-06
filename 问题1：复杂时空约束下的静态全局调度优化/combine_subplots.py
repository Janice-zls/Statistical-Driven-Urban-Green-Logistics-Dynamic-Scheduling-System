from PIL import Image
import os

def extract_subplots():
    # 路径定义
    base_dir = r"g:\B.比赛\2026第十八届华中杯大学生数学建模挑战赛\A题城市绿色物流配送调度_1776844160973\问题1"
    img1_path = os.path.join(base_dir, "图1_城市配送时空网络拓扑与需求特征.png")
    img2_path = os.path.join(base_dir, "图1_复合视窗_城市调度空间与容量联合分析面板.png")
    
    out_path = os.path.join(base_dir, "合并版_新2x2子图面板.png")
    
    # 打开图像
    img1 = Image.open(img1_path)
    img2 = Image.open(img2_path)
    
    # 获取尺寸
    w1, h1 = img1.size
    w2, h2 = img2.size
    
    # 裁剪 img1 的 (a) 和 (c) 子图 -> (左上) 和 (左下)
    # 因为标题可能占用顶部空间，我们直接二等分即可
    img1_a = img1.crop((0, 0, w1 // 2, h1 // 2))
    img1_c = img1.crop((0, h1 // 2, w1 // 2, h1))
    
    # 裁剪 img2 的 (c) 和 (d) 子图 -> (左下) 和 (右下)
    img2_c = img2.crop((0, h2 // 2, w2 // 2, h2))
    img2_d = img2.crop((w2 // 2, h2 // 2, w2, h2))
    
    # 统一尺寸，选取一个标准宽度和高度，例如取平均值或最大值
    # 这里我们为了保证清晰度，统一缩放为 2000x1500
    target_w = 2000
    target_h = 1500
    
    img1_a = img1_a.resize((target_w, target_h), Image.LANCZOS)
    img1_c = img1_c.resize((target_w, target_h), Image.LANCZOS)
    img2_c = img2_c.resize((target_w, target_h), Image.LANCZOS)
    img2_d = img2_d.resize((target_w, target_h), Image.LANCZOS)
    
    # 创建新的 2x2 画布
    # 宽度 = 2 * target_w, 高度 = 2 * target_h
    # 背景设为白色
    new_img = Image.new('RGB', (target_w * 2, target_h * 2), (255, 255, 255))
    
    # 按照用户要求拼接，这里假定排列顺序为：
    # 上半部分：img1_a, img1_c
    # 下半部分：img2_c, img2_d
    new_img.paste(img1_a, (0, 0))
    new_img.paste(img1_c, (target_w, 0))
    new_img.paste(img2_c, (0, target_h))
    new_img.paste(img2_d, (target_w, target_h))
    
    new_img.save(out_path, quality=95)
    print(f"合并成功！新图片保存在：{out_path}")

if __name__ == "__main__":
    extract_subplots()
