import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import json
from matplotlib.patches import FancyArrowPatch, Circle
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.patheffects as pe
import seaborn as sns
from matplotlib import cm
from matplotlib.colors import Normalize

def setup_premium_style():
    """设置极简、高级的顶级期刊/商业咨询风格（如McKinsey/Nature的极简排版）"""
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['Microsoft YaHei', 'SimHei', 'Arial'],
        'axes.unicode_minus': False,
        'axes.spines.top': False,
        'axes.spines.right': False,
        'axes.spines.left': False,
        'axes.spines.bottom': False, # 无边框设计
        'axes.grid': False,
        'axes.facecolor': '#F8F9FA', # 极浅的高级灰底色
        'figure.facecolor': '#F8F9FA',
        'font.size': 12,
        'axes.titlesize': 18,
        'axes.titleweight': 'bold',
        'legend.frameon': False,
        'figure.dpi': 300
    })

# 高端调色盘 (Cyan, Coral, Deep Blue, Mint, Gold)
PREMIUM_COLORS = ['#00B4D8', '#F07167', '#03045E', '#20B2AA', '#FFB703', '#9D4EDD']

def load_data():
    with open('问题1/processed_data.json', 'r', encoding='utf-8') as f:
        data = json.load(f)
    with open('问题1/solution_opt.json', 'r', encoding='utf-8') as f:
        sol = json.load(f)
    df_cost = pd.read_excel('问题1/车辆调度方案.xlsx')
    return data['nodes'], sol['routes'], df_cost

# ---------------------------------------------------------
# 图1: VRP问题中的终极杀器 —— 3D时空轨迹立方体 (Space-Time Cube)
# ---------------------------------------------------------
def plot_3d_space_time_cube(nodes, routes):
    setup_premium_style()
    fig = plt.figure(figsize=(14, 12))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_facecolor('#F8F9FA')
    
    # 绘制底部的2D节点映射
    x_coords = [n['x'] for n in nodes if n['id'] != 0]
    y_coords = [n['y'] for n in nodes if n['id'] != 0]
    ax.scatter(x_coords, y_coords, np.zeros_like(x_coords)+8, c='gray', s=10, alpha=0.2)
    ax.scatter(0, 0, 8, c='#F07167', s=200, marker='*') # 配送中心
    
    # 抽取3条经典路线，绘制3D螺旋上升的时空轨迹
    np.random.seed(42)
    sample_routes = np.random.choice(routes, 4, replace=False)
    
    for idx, r in enumerate(sample_routes):
        route = r['route']
        t_start = r.get('start_time', 8.0)
        xs, ys, ts = [], [], []
        curr_t = t_start
        
        for i, n_id in enumerate(route):
            node = nodes[n_id if isinstance(n_id, int) else int(str(n_id).split('_')[0])]
            xs.append(node['x'])
            ys.append(node['y'])
            
            # 简化的时间推算（仅用于可视化，非精确）
            if i > 0:
                prev_n = nodes[route[i-1] if isinstance(route[i-1], int) else int(str(route[i-1]).split('_')[0])]
                dist = np.sqrt((node['x']-prev_n['x'])**2 + (node['y']-prev_n['y'])**2)
                curr_t += (dist / 35.0) # 假设35km/h
                if n_id != 0:
                    curr_t = max(curr_t, node['tw_start'])
                    curr_t += (20/60.0) # 服务时间
            ts.append(curr_t)
            
            # 在该节点画一个时间窗的柱子 (从 tw_start 到 tw_end)
            if n_id != 0 and i % 2 == 0: # 抽样画柱子避免太乱
                ax.plot([node['x'], node['x']], [node['y'], node['y']], [node['tw_start'], node['tw_end']], 
                        color='gray', alpha=0.3, lw=2)
        
        # 绘制3D轨迹线，带发光阴影效果
        color = PREMIUM_COLORS[idx % len(PREMIUM_COLORS)]
        ax.plot(xs, ys, ts, color=color, lw=3, alpha=0.9, label=f'调度轨迹 {idx+1}',
                path_effects=[pe.SimpleLineShadow(shadow_color='black', alpha=0.1, offset=(1, -1)), pe.Normal()])
        # 绘制节点气泡
        ax.scatter(xs, ys, ts, color=color, s=50, edgecolors='white', alpha=1.0, zorder=5)

    ax.set_title('图1：物流配送网络3D时空演化轨迹 (Space-Time Cube)', pad=30)
    ax.set_xlabel('\n\n空间 X (km)', labelpad=20)
    ax.set_ylabel('\n\n空间 Y (km)', labelpad=20)
    ax.set_zlabel('\n\n时间轴 (24小时制)', labelpad=20)
    ax.set_zlim(8, 20)
    ax.view_init(elev=25, azim=-45) # 绝佳视角
    ax.legend(loc='upper left', bbox_to_anchor=(1.1, 0.9))
    
    # 隐藏3D网格线以提升高级感
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.xaxis.pane.set_edgecolor('w')
    ax.yaxis.pane.set_edgecolor('w')
    ax.zaxis.pane.set_edgecolor('w')
    ax.grid(color='white', linestyle='-', linewidth=1.5, alpha=0.5)

    plt.savefig('问题1/高级视效_1_3D时空演化轨迹.png', bbox_inches='tight', facecolor='#F8F9FA')
    plt.savefig('问题1/高级视效_1_3D时空演化轨迹.pdf', bbox_inches='tight', facecolor='#F8F9FA')
    plt.close()

# ---------------------------------------------------------
# 图2: 动态贝塞尔流向网络图 (Curved Network Flow)
# ---------------------------------------------------------
def plot_bezier_network(nodes, routes):
    setup_premium_style()
    fig, ax = plt.subplots(figsize=(14, 14))
    
    # 绘制背景绿色配送区
    theta = np.linspace(0, 2*np.pi, 100)
    ax.plot(10*np.cos(theta), 10*np.sin(theta), color='#20B2AA', linestyle='--', lw=2)
    ax.fill_between(10*np.cos(theta), 10*np.sin(theta), color='#20B2AA', alpha=0.05)
    
    # 绘制节点 (大小代表需求重量，颜色代表时间窗早晚)
    w_list = [n['demand_w'] for n in nodes if n['id']!=0]
    t_list = [n['tw_start'] for n in nodes if n['id']!=0]
    norm = Normalize(vmin=8, vmax=18)
    cmap = cm.get_cmap('plasma_r')
    
    for n in nodes:
        if n['id'] == 0: continue
        color = cmap(norm(n['tw_start']))
        size = (n['demand_w'] / 3000.0) * 300 + 20
        ax.scatter(n['x'], n['y'], s=size, color=color, alpha=0.7, edgecolors='white', lw=1, zorder=3)
        
    ax.scatter(0, 0, s=800, c='#F07167', marker='p', edgecolors='white', lw=3, zorder=5, label='配送中心')
    
    # 抽取6条路线绘制高级贝塞尔曲线
    np.random.seed(123)
    sample_routes = np.random.choice(routes, 6, replace=False)
    
    for idx, r in enumerate(sample_routes):
        route = r['route']
        color = PREMIUM_COLORS[idx % len(PREMIUM_COLORS)]
        
        for i in range(len(route)-1):
            u = nodes[route[i] if isinstance(route[i], int) else int(str(route[i]).split('_')[0])]
            v = nodes[route[i+1] if isinstance(route[i+1], int) else int(str(route[i+1]).split('_')[0])]
            
            # 使用 FancyArrowPatch 绘制优美的弧线 (arc3)
            arrow = FancyArrowPatch((u['x'], u['y']), (v['x'], v['y']),
                                    connectionstyle="arc3,rad=0.2",
                                    color=color,
                                    alpha=0.6,
                                    lw=2.5,
                                    arrowstyle='-|>',
                                    mutation_scale=15,
                                    zorder=2)
            ax.add_patch(arrow)
            
    ax.set_title('图2：物流节点动态需求与非线性流向网络拓扑图', pad=20)
    ax.set_aspect('equal')
    ax.set_xticks([]) # 隐藏坐标轴
    ax.set_yticks([])
    
    # 增加颜色条 (Colorbar)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, shrink=0.6, pad=0.02)
    cbar.set_label('客户最早服务时间 (24h)', rotation=270, labelpad=20)
    
    plt.savefig('问题1/高级视效_2_贝塞尔非线性流向网络.png', bbox_inches='tight', facecolor='#F8F9FA')
    plt.savefig('问题1/高级视效_2_贝塞尔非线性流向网络.pdf', bbox_inches='tight', facecolor='#F8F9FA')
    plt.close()

# ---------------------------------------------------------
# 图3: 南丁格尔玫瑰图 (Nightingale Rose Chart / Polar Bar)
# ---------------------------------------------------------
def plot_nightingale_rose(df_cost):
    setup_premium_style()
    fig = plt.figure(figsize=(12, 12))
    ax = fig.add_subplot(111, polar=True)
    
    # 将车次按照总成本分为8个区间，统计每个区间的数量，用来展示成本分布
    bins = np.linspace(df_cost['总成本(元)'].min(), df_cost['总成本(元)'].max(), 12)
    hist, _ = np.histogram(df_cost['总成本(元)'], bins=bins)
    
    # 数据准备
    N = len(hist)
    theta = np.linspace(0.0, 2 * np.pi, N, endpoint=False)
    radii = hist
    width = 2 * np.pi / N
    
    # 使用高级渐变色
    colors = plt.cm.viridis(radii / max(radii))
    
    bars = ax.bar(theta, radii, width=width, bottom=0.0, color=colors, alpha=0.8, edgecolor='white', lw=2)
    
    # 隐藏极坐标轴的网格和刻度，提升逼格
    ax.set_yticklabels([])
    ax.set_xticks(theta)
    
    # 设置刻度标签为成本区间
    labels = [f"¥{bins[i]:.0f}-{bins[i+1]:.0f}" for i in range(N)]
    ax.set_xticklabels(labels, fontsize=10, rotation=45)
    ax.spines['polar'].set_visible(False) # 隐藏最外层圆圈
    ax.grid(color='#E0E0E0', linestyle=':', linewidth=1)
    
    # 中心留白 (类似于甜甜圈玫瑰图)
    ax.set_rorigin(-max(radii)*0.2)
    
    ax.set_title('图3：全量调度单次综合成本分布频率 (南丁格尔玫瑰图)', pad=40)
    
    plt.savefig('问题1/高级视效_3_南丁格尔玫瑰图.png', bbox_inches='tight', facecolor='#F8F9FA')
    plt.savefig('问题1/高级视效_3_南丁格尔玫瑰图.pdf', bbox_inches='tight', facecolor='#F8F9FA')
    plt.close()

# ---------------------------------------------------------
# 图4: 六边形网格空间热力图 (Hexbin Spatial Density)
# ---------------------------------------------------------
def plot_hexbin_density(nodes):
    setup_premium_style()
    fig, ax = plt.subplots(figsize=(14, 12))
    
    x = [n['x'] for n in nodes if n['id']!=0]
    y = [n['y'] for n in nodes if n['id']!=0]
    w = [n['demand_w'] for n in nodes if n['id']!=0]
    
    # 绘制高大上的六边形网格 (Hexbin)，C 参数为重量，使用均值聚合
    hb = ax.hexbin(x, y, C=w, gridsize=15, cmap='magma_r', edgecolors='white', linewidths=1.5, reduce_C_function=np.sum, alpha=0.9)
    
    # 添加配送中心
    ax.scatter(0, 0, c='cyan', s=400, marker='*', edgecolors='white', lw=2, zorder=5, label='配送中心')
    
    # 添加等高线叠加层，极具数据科学感
    sns.kdeplot(x=x, y=y, levels=5, color='#03045E', linewidths=1.5, alpha=0.5, ax=ax)
    
    ax.set_title('图4：城市绿色物流需求空间重力蜂巢图 (Hexbin Aggregation)', pad=20)
    ax.set_aspect('equal')
    ax.set_xticks([])
    ax.set_yticks([])
    
    cb = fig.colorbar(hb, ax=ax, shrink=0.7, pad=0.02)
    cb.set_label('该区域累计需求重量 (kg)', rotation=270, labelpad=20)
    
    plt.savefig('问题1/高级视效_4_蜂巢空间热力图.png', bbox_inches='tight', facecolor='#F8F9FA')
    plt.savefig('问题1/高级视效_4_蜂巢空间热力图.pdf', bbox_inches='tight', facecolor='#F8F9FA')
    plt.close()

if __name__ == '__main__':
    nodes, routes, df_cost = load_data()
    plot_3d_space_time_cube(nodes, routes)
    plot_bezier_network(nodes, routes)
    plot_nightingale_rose(df_cost)
    plot_hexbin_density(nodes)
    print("【耳目一新】级别可视化已生成！")