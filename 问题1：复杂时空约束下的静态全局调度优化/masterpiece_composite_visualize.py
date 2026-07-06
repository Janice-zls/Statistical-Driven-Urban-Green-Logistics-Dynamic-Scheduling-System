import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import json
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.patheffects as pe
import seaborn as sns
from matplotlib.colors import Normalize
import matplotlib.cm as cm
import warnings

warnings.filterwarnings('ignore')

def setup_premium_style():
    """设置顶级期刊/商业咨询风格（如McKinsey/Nature的极简排版）"""
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['Microsoft YaHei', 'SimHei', 'Arial'],
        'axes.unicode_minus': False,
        'axes.spines.top': False,
        'axes.spines.right': False,
        'axes.spines.left': False,
        'axes.spines.bottom': False,
        'axes.grid': False,
        'axes.facecolor': '#F8F9FA',
        'figure.facecolor': '#F8F9FA',
        'font.size': 12,
        'axes.titlesize': 18,
        'axes.titleweight': 'bold',
        'legend.frameon': False,
        'figure.dpi': 300
    })

PREMIUM_COLORS = ['#00B4D8', '#F07167', '#03045E', '#20B2AA', '#FFB703', '#9D4EDD']

def load_data():
    with open('问题1/processed_data.json', 'r', encoding='utf-8') as f:
        data = json.load(f)
    with open('问题1/solution_opt.json', 'r', encoding='utf-8') as f:
        sol = json.load(f)
    df_cost = pd.read_excel('问题1/车辆调度方案.xlsx')
    return data['nodes'], sol['routes'], df_cost

def plot_ultimate_masterpiece(nodes, routes, df_cost):
    setup_premium_style()
    # 建立巨型画布
    fig = plt.figure(figsize=(24, 22))
    
    # 采用GridSpec构建2x2的高端排版面板，适当留白呼吸感
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.35, wspace=0.25)
    fig.suptitle("A题：城市绿色物流时空演化与调度综合分析面板", fontsize=32, fontweight='bold', y=0.96, color='#111111')

    # =========================================================
    # (a) 3D时空演化 (Space-Time Cube) - 左上角
    # =========================================================
    ax1 = fig.add_subplot(gs[0, 0], projection='3d')
    ax1.set_facecolor('#F8F9FA')
    x_coords = [n['x'] for n in nodes if n['id'] != 0]
    y_coords = [n['y'] for n in nodes if n['id'] != 0]
    ax1.scatter(x_coords, y_coords, np.zeros_like(x_coords)+8, c='gray', s=20, alpha=0.15)
    ax1.scatter(0, 0, 8, c='#F07167', s=400, marker='*', edgecolors='white', lw=1)
    
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
            if i > 0:
                prev_n = nodes[route[i-1] if isinstance(route[i-1], int) else int(str(route[i-1]).split('_')[0])]
                dist = np.sqrt((node['x']-prev_n['x'])**2 + (node['y']-prev_n['y'])**2)
                curr_t += (dist / 35.0)
                if n_id != 0:
                    curr_t = max(curr_t, node['tw_start'])
                    curr_t += (20/60.0)
            ts.append(curr_t)
            # 画少量时间窗柱子避免杂乱
            if n_id != 0 and i % 3 == 0:
                ax1.plot([node['x'], node['x']], [node['y'], node['y']], [node['tw_start'], node['tw_end']], 
                        color='gray', alpha=0.3, lw=2)
        
        color = PREMIUM_COLORS[idx % len(PREMIUM_COLORS)]
        # 带有发光阴影效果的3D轨迹
        ax1.plot(xs, ys, ts, color=color, lw=3.5, alpha=0.9,
                path_effects=[pe.SimpleLineShadow(shadow_color='black', alpha=0.1, offset=(1, -1)), pe.Normal()])
        ax1.scatter(xs, ys, ts, color=color, s=70, edgecolors='white', alpha=1.0, zorder=5)

    ax1.set_title('(a) VRP 3D时空螺旋轨迹演化图 (Space-Time Cube)', pad=25)
    ax1.set_xlabel('\n空间 X (km)', labelpad=15)
    ax1.set_ylabel('\n空间 Y (km)', labelpad=15)
    ax1.set_zlabel('\n时间轴 (24h)', labelpad=15)
    ax1.set_zlim(8, 20)
    ax1.view_init(elev=22, azim=-55)
    # 隐藏多余3D网格与背景
    ax1.xaxis.pane.fill = False
    ax1.yaxis.pane.fill = False
    ax1.zaxis.pane.fill = False
    ax1.grid(color='white', linestyle='-', linewidth=1.5, alpha=0.5)

    # =========================================================
    # (b) 贝塞尔网络流 (Bezier Curved Flow) - 右上角
    # =========================================================
    ax2 = fig.add_subplot(gs[0, 1])
    # 绿色配送区
    theta = np.linspace(0, 2*np.pi, 100)
    ax2.plot(10*np.cos(theta), 10*np.sin(theta), color='#20B2AA', linestyle='--', lw=2)
    ax2.fill_between(10*np.cos(theta), 10*np.sin(theta), color='#20B2AA', alpha=0.05)
    
    norm_c = Normalize(vmin=8, vmax=18)
    cmap_bezier = cm.plasma_r
    
    for n in nodes:
        if n['id'] == 0: continue
        color = cmap_bezier(norm_c(n['tw_start']))
        size = (n['demand_w'] / 3000.0) * 500 + 40
        ax2.scatter(n['x'], n['y'], s=size, color=color, alpha=0.8, edgecolors='white', lw=1.5, zorder=3)
    ax2.scatter(0, 0, s=1000, c='#F07167', marker='p', edgecolors='white', lw=3, zorder=5)
    
    np.random.seed(123)
    sample_routes2 = np.random.choice(routes, 7, replace=False)
    for idx, r in enumerate(sample_routes2):
        route = r['route']
        color = PREMIUM_COLORS[idx % len(PREMIUM_COLORS)]
        for i in range(len(route)-1):
            u = nodes[route[i] if isinstance(route[i], int) else int(str(route[i]).split('_')[0])]
            v = nodes[route[i+1] if isinstance(route[i+1], int) else int(str(route[i+1]).split('_')[0])]
            arrow = FancyArrowPatch((u['x'], u['y']), (v['x'], v['y']),
                                    connectionstyle="arc3,rad=0.25", color=color, alpha=0.6, lw=3,
                                    arrowstyle='-|>', mutation_scale=25, zorder=2)
            ax2.add_patch(arrow)
            
    ax2.set_title('(b) 动态贝塞尔非线性流向网络拓扑图', pad=25)
    ax2.set_aspect('equal')
    ax2.set_xticks([])
    ax2.set_yticks([])
    
    # 将色带放在下方避免拥挤
    sm = plt.cm.ScalarMappable(cmap=cmap_bezier, norm=norm_c)
    sm.set_array([])
    cbar2 = fig.colorbar(sm, ax=ax2, orientation='horizontal', shrink=0.6, pad=0.05)
    cbar2.set_label('客户最早服务时间窗 (24h制)')

    # =========================================================
    # (c) 六边形蜂巢密度热力图 (Hexbin Aggregation) - 左下角
    # =========================================================
    ax3 = fig.add_subplot(gs[1, 0])
    x = [n['x'] for n in nodes if n['id']!=0]
    y = [n['y'] for n in nodes if n['id']!=0]
    w = [n['demand_w'] for n in nodes if n['id']!=0]
    
    hb = ax3.hexbin(x, y, C=w, gridsize=18, cmap='magma_r', edgecolors='white', linewidths=2, reduce_C_function=np.sum, alpha=0.95)
    ax3.scatter(0, 0, c='cyan', s=600, marker='*', edgecolors='white', lw=2.5, zorder=5)
    sns.kdeplot(x=x, y=y, levels=7, color='#03045E', linewidths=2, alpha=0.6, ax=ax3)
    
    ax3.set_title('(c) 城市物流需求空间重力蜂巢聚合图 (Hexbin & KDE)', pad=25)
    ax3.set_aspect('equal')
    ax3.set_xticks([])
    ax3.set_yticks([])
    
    cbar3 = fig.colorbar(hb, ax=ax3, orientation='horizontal', shrink=0.6, pad=0.05)
    cbar3.set_label('该蜂巢区域累计需求重量 (kg)')

    # =========================================================
    # (d) 调度成本频率 南丁格尔玫瑰图 (Nightingale Rose) - 右下角
    # =========================================================
    ax4 = fig.add_subplot(gs[1, 1], polar=True)
    bins = np.linspace(df_cost['总成本(元)'].min(), df_cost['总成本(元)'].max(), 14)
    hist, _ = np.histogram(df_cost['总成本(元)'], bins=bins)
    
    N = len(hist)
    theta_polar = np.linspace(0.0, 2 * np.pi, N, endpoint=False)
    radii = hist
    width = 2 * np.pi / N
    colors_polar = plt.cm.viridis(radii / max(radii))
    
    ax4.bar(theta_polar, radii, width=width, bottom=0.0, color=colors_polar, alpha=0.9, edgecolor='white', lw=3)
    ax4.set_yticklabels([])
    ax4.set_xticks(theta_polar)
    
    labels = [f"¥{bins[i]:.0f}" for i in range(N)]
    ax4.set_xticklabels(labels, fontsize=11, rotation=45, color='#333333')
    ax4.spines['polar'].set_visible(False)
    ax4.grid(color='#E0E0E0', linestyle='--', linewidth=1.5)
    ax4.set_rorigin(-max(radii)*0.25) # 玫瑰图中心甜甜圈效果留白
    
    ax4.set_title('(d) 单次调度总成本分布频率 (Nightingale Rose Chart)', pad=40)

    # 保存
    plt.savefig('问题1/最终成果_极简高级组合大屏面板.png', bbox_inches='tight', facecolor='#F8F9FA')
    plt.savefig('问题1/最终成果_极简高级组合大屏面板.pdf', bbox_inches='tight', facecolor='#F8F9FA')
    plt.close()

if __name__ == '__main__':
    nodes, routes, df_cost = load_data()
    plot_ultimate_masterpiece(nodes, routes, df_cost)
    print("【完美结合】包含3D轨迹、贝塞尔网络、蜂巢热力图与玫瑰图的复合大屏面板已生成！")