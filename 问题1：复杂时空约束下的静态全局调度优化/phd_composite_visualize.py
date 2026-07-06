import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import json
import numpy as np
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

# ----------------- 1. 全局样式设置 (SCI/Nature 级别) -----------------
def setup_phd_style():
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['SimHei', 'Microsoft YaHei', 'Arial'],
        'axes.unicode_minus': False,
        'axes.spines.top': False,
        'axes.spines.right': False,
        'axes.linewidth': 1.5,
        'xtick.major.width': 1.5,
        'ytick.major.width': 1.5,
        'font.size': 12,
        'axes.titlesize': 15,
        'axes.titleweight': 'bold',
        'axes.labelsize': 13,
        'legend.fontsize': 11,
        'legend.frameon': False,
        'figure.dpi': 300,
        'figure.facecolor': 'white'
    })

# NPG (Nature Publishing Group) Palette
NPG_COLORS = ['#E64B35', '#4DBBD5', '#00A087', '#3C5488', '#F39B7F', '#8491B4', '#91D1C2', '#DC0000', '#7E6148', '#B09C85']

# ----------------- 2. 数据读取 -----------------
def load_data():
    with open('问题1/processed_data.json', 'r', encoding='utf-8') as f:
        data = json.load(f)
    with open('问题1/solution_opt.json', 'r', encoding='utf-8') as f:
        sol = json.load(f)
    df_cost = pd.read_excel('问题1/车辆调度方案.xlsx')
    return data['nodes'], sol['routes'], df_cost

# ----------------- 3. 绘制复合视窗 1 -----------------
def plot_composite_figure_1(nodes, routes, df_cost):
    """
    复合图1：时空分布与需求容量多维联合分析面板
    (a) 城市配送网络空间路径与核心区放大
    (b) 客户需求分布（重量 vs 体积）双变量核密度估计 (Bivariate KDE)
    (c) 车辆双维利用率散点图与最优装载区间
    (d) 配送出发时间分布统计 (KDE)
    """
    setup_phd_style()
    fig = plt.figure(figsize=(18, 14))
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.3, wspace=0.25)
    
    # ---- (a) 空间分布与核心区放大 ----
    ax1 = fig.add_subplot(gs[0, 0])
    
    # 绘制绿色配送区
    theta = np.linspace(0, 2*np.pi, 100)
    ax1.plot(10*np.cos(theta), 10*np.sin(theta), color='#00A087', linestyle='--', linewidth=2, label='核心区(r=10km)', zorder=1)
    ax1.fill_between(10*np.cos(theta), 10*np.sin(theta), color='#00A087', alpha=0.05, zorder=0)
    
    # 绘制节点
    x_coords = [n['x'] for n in nodes if n['id'] != 0]
    y_coords = [n['y'] for n in nodes if n['id'] != 0]
    ax1.scatter(x_coords, y_coords, c='gray', s=15, alpha=0.5, label='客户点', zorder=2)
    ax1.scatter(0, 0, c='#E64B35', marker='*', s=300, edgecolor='white', label='配送中心', zorder=4)
    
    # 随机挑选5条路线绘制（避免过密）
    np.random.seed(42)
    sample_routes = np.random.choice(routes, min(5, len(routes)), replace=False)
    for i, r in enumerate(sample_routes):
        rx = [nodes[nid]['x'] for n_id in r['route'] for nid in [n_id if isinstance(n_id, int) else int(n_id.split('_')[0])]]
        ry = [nodes[nid]['y'] for n_id in r['route'] for nid in [n_id if isinstance(n_id, int) else int(n_id.split('_')[0])]]
        ax1.plot(rx, ry, color=NPG_COLORS[i%len(NPG_COLORS)], linewidth=1.5, alpha=0.7, marker='o', markersize=3, 
                 label=f'抽样路线 {i+1}')
        
    ax1.set_title('(a) 城市配送网络空间路径与抽样调度轨迹', loc='left')
    ax1.set_xlabel('X 坐标 (km)')
    ax1.set_ylabel('Y 坐标 (km)')
    ax1.legend(loc='upper right', bbox_to_anchor=(1.25, 1))
    
    # 局部放大图 (Inset)
    axins = inset_axes(ax1, width="35%", height="35%", loc='lower left', borderpad=2)
    axins.plot(10*np.cos(theta), 10*np.sin(theta), color='#00A087', linestyle='--', linewidth=1)
    axins.fill_between(10*np.cos(theta), 10*np.sin(theta), color='#00A087', alpha=0.05)
    axins.scatter(x_coords, y_coords, c='gray', s=20, alpha=0.6)
    axins.scatter(0, 0, c='#E64B35', marker='*', s=200, edgecolor='white')
    for i, r in enumerate(sample_routes):
        rx = [nodes[nid]['x'] for n_id in r['route'] for nid in [n_id if isinstance(n_id, int) else int(n_id.split('_')[0])]]
        ry = [nodes[nid]['y'] for n_id in r['route'] for nid in [n_id if isinstance(n_id, int) else int(n_id.split('_')[0])]]
        axins.plot(rx, ry, color=NPG_COLORS[i%len(NPG_COLORS)], linewidth=1.5, alpha=0.8)
    axins.set_xlim(-12, 12)
    axins.set_ylim(-12, 12)
    axins.set_xticks([])
    axins.set_yticks([])
    axins.set_title('核心区放大', fontsize=10)
    for spine in axins.spines.values():
        spine.set_edgecolor('#00A087')
        spine.set_linewidth(1.5)
        
    # ---- (b) 需求双变量 KDE 分布 ----
    ax2 = fig.add_subplot(gs[0, 1])
    w_list = [n['demand_w'] for n in nodes if n['id'] != 0]
    v_list = [n['demand_v'] for n in nodes if n['id'] != 0]
    sns.kdeplot(x=w_list, y=v_list, cmap="mako", fill=True, bw_adjust=.5, ax=ax2)
    sns.scatterplot(x=w_list, y=v_list, s=15, color=".15", alpha=0.3, ax=ax2)
    ax2.set_title('(b) 客户需求规模(重量-体积)双变量核密度估计', loc='left')
    ax2.set_xlabel('客户需求重量 (kg)')
    ax2.set_ylabel('客户需求体积 (m³)')
    ax2.set_xlim(0, max(w_list)*1.1)
    ax2.set_ylim(0, max(v_list)*1.1)

    # ---- (c) 装载率与容积利用率联合散点图 ----
    ax3 = fig.add_subplot(gs[1, 0])
    utilization = []
    # 所有车现在都是类型4(载重3000，容积15.0)
    for r in routes:
        total_w = sum(d['w'] for d in r['demands'])
        total_v = sum(d['v'] for d in r['demands'])
        w_rate = total_w / 3000.0 * 100
        v_rate = total_v / 15.0 * 100
        utilization.append({'w_rate': w_rate, 'v_rate': v_rate})
        
    df_util = pd.DataFrame(utilization)
    sns.scatterplot(data=df_util, x='w_rate', y='v_rate', s=100, color=NPG_COLORS[3], alpha=0.7, edgecolor='white', ax=ax3)
    ax3.axhline(100, color='gray', linestyle='--', alpha=0.5)
    ax3.axvline(100, color='gray', linestyle='--', alpha=0.5)
    ax3.fill_between([80, 100], 80, 100, color=NPG_COLORS[6], alpha=0.15, label='高利用率区间 (80%-100%)')
    ax3.set_title('(c) 车辆单趟调度装载率与容积利用率联合评估', loc='left')
    ax3.set_xlabel('车辆载重利用率 (%)')
    ax3.set_ylabel('车辆容积利用率 (%)')
    ax3.set_xlim(0, 105)
    ax3.set_ylim(0, 105)
    ax3.legend(loc='lower left')

    # ---- (d) 出发时间分布 ----
    ax4 = fig.add_subplot(gs[1, 1])
    start_times = [r.get('start_time', 8.0) for r in routes]
    sns.histplot(start_times, bins=20, kde=True, color=NPG_COLORS[4], ax=ax4, alpha=0.6, line_kws={'linewidth': 2})
    ax4.set_title('(d) 车辆智能延迟出发时间策略密度分布', loc='left')
    ax4.set_xlabel('车辆出发时间 (24小时制)')
    ax4.set_ylabel('出车频次 (辆次)')
    ax4.set_xticks(range(0, 25, 2))
    
    # 调整并保存
    plt.tight_layout()
    plt.savefig('问题1/图1_复合视窗_城市调度空间与容量联合分析面板.png', bbox_inches='tight')
    plt.savefig('问题1/图1_复合视窗_城市调度空间与容量联合分析面板.pdf', bbox_inches='tight')
    plt.close()

# ----------------- 4. 绘制复合视窗 2 -----------------
def plot_composite_figure_2(df_cost):
    """
    复合图2：多维成本分析与经济性特征面板
    (a) 各子项成本构成箱型图 (Boxplot)
    (b) 单车趟次总成本云雨图 (Raincloud / Violin+Strip)
    (c) 惩罚成本分布直方图
    (d) 车辆出车时间与总成本相关性散点图 (假设时间数据可以从df_cost提取)
    """
    setup_phd_style()
    fig = plt.figure(figsize=(18, 12))
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.3, wspace=0.25)
    
    # 提取时间
    # "0(10:30) -> ..." 提取10:30
    def extract_start_hour(t_str):
        first_node = t_str.split('->')[0].strip()
        time_part = first_node.split('(')[1].replace(')', '')
        h, m = map(int, time_part.split(':'))
        return h + m/60.0
    
    df_cost['出发时间'] = df_cost['到达时间节点'].apply(extract_start_hour)
    
    # ---- (a) 各子项成本箱线图 ----
    ax1 = fig.add_subplot(gs[0, 0])
    cost_melted = pd.melt(df_cost, value_vars=['固定成本(元)', '行驶与碳排放成本(元)', '时间窗惩罚成本(元)'], 
                          var_name='成本类别', value_name='金额(元)')
    sns.boxplot(data=cost_melted, x='成本类别', y='金额(元)', palette=NPG_COLORS[1:4], width=0.5, ax=ax1, showfliers=False)
    sns.stripplot(data=cost_melted, x='成本类别', y='金额(元)', color=".25", size=4, alpha=0.4, ax=ax1, jitter=True)
    ax1.set_title('(a) 单趟调度各分项成本微观统计学分布', loc='left')
    ax1.set_xlabel('')
    ax1.set_ylabel('单趟金额 (元)')
    
    # ---- (b) 云雨图: 单车趟次总成本分布 ----
    ax2 = fig.add_subplot(gs[0, 1])
    # 由于只有一个车型，我们直接画所有车型的分布
    sns.violinplot(data=df_cost, y='总成本(元)', color=NPG_COLORS[0], inner="quartile", alpha=0.5, ax=ax2, bw_adjust=0.5)
    sns.stripplot(data=df_cost, y='总成本(元)', color=".25", size=5, alpha=0.5, ax=ax2, jitter=0.05)
    ax2.set_title('(b) 单趟车辆调度综合成本密度特征 (云雨图)', loc='left')
    ax2.set_ylabel('单趟综合成本 (元)')
    
    # ---- (c) 行驶路径长度(节点数) vs 行驶成本 ----
    ax3 = fig.add_subplot(gs[1, 0])
    df_cost['途经客户数'] = df_cost['行驶路径'].apply(lambda x: len(x.split('->')) - 2)
    sns.regplot(data=df_cost, x='途经客户数', y='行驶与碳排放成本(元)', color=NPG_COLORS[2], scatter_kws={'s': 50, 'alpha': 0.6}, line_kws={'linewidth': 2}, ax=ax3)
    ax3.set_title('(c) 车辆单趟途经客户数与行驶成本回归分析', loc='left')
    ax3.set_xlabel('单趟途经客户数 (个)')
    ax3.set_ylabel('行驶与碳排放成本 (元)')
    
    # ---- (d) 出发时间与惩罚成本的联合分布 ----
    ax4 = fig.add_subplot(gs[1, 1])
    sns.scatterplot(data=df_cost, x='出发时间', y='时间窗惩罚成本(元)', size='总成本(元)', sizes=(20, 200), 
                    color=NPG_COLORS[7], alpha=0.7, ax=ax4)
    ax4.set_title('(d) 动态出发时间与时间窗惩罚成本联合图 (Bubble)', loc='left')
    ax4.set_xlabel('出发时间 (24小时制)')
    ax4.set_ylabel('时间窗惩罚成本 (元)')
    ax4.set_xticks(range(0, 25, 2))
    ax4.legend(bbox_to_anchor=(1.05, 1), loc='upper left', title='总成本')

    plt.tight_layout()
    plt.savefig('问题1/图2_复合视窗_多维成本分析与经济性特征面板.png', bbox_inches='tight')
    plt.savefig('问题1/图2_复合视窗_多维成本分析与经济性特征面板.pdf', bbox_inches='tight')
    plt.close()

if __name__ == '__main__':
    nodes, routes, df_cost = load_data()
    plot_composite_figure_1(nodes, routes, df_cost)
    plot_composite_figure_2(df_cost)
    print("博士级高密度复合学术图表（GridSpec组图）生成完毕！")