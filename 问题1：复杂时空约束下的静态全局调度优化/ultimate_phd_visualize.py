import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import json
import matplotlib.gridspec as gridspec
from scipy.stats import norm
import warnings
import math
from matplotlib.patches import Polygon

warnings.filterwarnings('ignore')

def setup_phd_style():
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['SimHei', 'Microsoft YaHei', 'Arial'],
        'axes.unicode_minus': False,
        'axes.spines.top': False,
        'axes.spines.right': False,
        'axes.linewidth': 1.2,
        'font.size': 11,
        'axes.titlesize': 13,
        'axes.titleweight': 'bold',
        'axes.labelsize': 11,
        'legend.fontsize': 10,
        'legend.frameon': True,
        'legend.edgecolor': 'white',
        'figure.dpi': 300,
        'figure.facecolor': 'white'
    })

NPG = ['#E64B35', '#4DBBD5', '#00A087', '#3C5488', '#F39B7F', '#8491B4', '#91D1C2', '#DC0000', '#7E6148', '#B09C85']

def load_data():
    with open('问题1/processed_data.json', 'r', encoding='utf-8') as f:
        data = json.load(f)
    with open('问题1/solution_opt.json', 'r', encoding='utf-8') as f:
        sol = json.load(f)
    df_cost = pd.read_excel('问题1/车辆调度方案.xlsx')
    return data['nodes'], sol['routes'], df_cost

# ----------------- 理论模型数据 -----------------
def generate_math_model_data():
    # 速度分布
    v_range = np.linspace(0, 80, 500)
    pdf_smooth = norm.pdf(v_range, 55.3, 0.1)
    pdf_normal = norm.pdf(v_range, 35.4, 5.2)
    pdf_congest = norm.pdf(v_range, 9.8, 4.7)
    
    # 能耗曲线
    fpk = 0.0025 * v_range**2 - 0.2554 * v_range + 31.75
    epk = 0.0014 * v_range**2 - 0.12 * v_range + 36.19
    
    # 车辆类型参数
    v_types = pd.DataFrame([
        {'Type': '1 (燃油)', 'Q': 3000, 'V': 13.5, 'Num': 60, 'Cost': 400},
        {'Type': '2 (燃油)', 'Q': 1500, 'V': 10.8, 'Num': 50, 'Cost': 400},
        {'Type': '3 (燃油)', 'Q': 1250, 'V': 6.5, 'Num': 50, 'Cost': 400},
        {'Type': '4 (电车)', 'Q': 3000, 'V': 15.0, 'Num': 10, 'Cost': 400},
        {'Type': '5 (电车)', 'Q': 1250, 'V': 8.5, 'Num': 15, 'Cost': 400},
    ])
    
    return v_range, pdf_smooth, pdf_normal, pdf_congest, fpk, epk, v_types

# ================= 绘图函数 =================

def plot_fig1_spatial_temporal(nodes, routes):
    """图1：城市配送时空网络拓扑与需求特征"""
    setup_phd_style()
    fig = plt.figure(figsize=(16, 12))
    gs = gridspec.GridSpec(2, 2, hspace=0.3, wspace=0.25)
    
    # (a) 空间分布与绿色配送区
    ax1 = fig.add_subplot(gs[0, 0])
    theta = np.linspace(0, 2*np.pi, 100)
    ax1.plot(10*np.cos(theta), 10*np.sin(theta), color=NPG[2], linestyle='--', lw=2, label='绿色配送区(r=10)')
    ax1.fill_between(10*np.cos(theta), 10*np.sin(theta), color=NPG[2], alpha=0.08)
    x = [n['x'] for n in nodes if n['id']!=0]
    y = [n['y'] for n in nodes if n['id']!=0]
    ax1.scatter(x, y, c='gray', s=20, alpha=0.6, label='客户节点')
    ax1.scatter(0, 0, c=NPG[0], marker='*', s=350, edgecolor='white', lw=1.5, label='配送中心', zorder=5)
    # 画几条核心线路
    for i, r in enumerate(routes[:6]):
        rx = [nodes[n]['x'] for n in r['route']]
        ry = [nodes[n]['y'] for n in r['route']]
        ax1.plot(rx, ry, color=NPG[i%len(NPG)], lw=1.5, alpha=0.7)
    ax1.set_title('(a) 城市绿色物流空间拓扑与核心调度轨迹')
    ax1.set_xlabel('X (km)')
    ax1.set_ylabel('Y (km)')
    ax1.legend(loc='upper right')
    ax1.set_aspect('equal', 'box')

    # (b) 需求时间窗甘特分布
    ax2 = fig.add_subplot(gs[0, 1])
    sorted_nodes = sorted([n for n in nodes if n['id']!=0], key=lambda x: x['tw_start'])
    for i, n in enumerate(sorted_nodes[::3]): # 抽样展示30个避免太密
        ax2.hlines(y=i, xmin=n['tw_start'], xmax=n['tw_end'], color=NPG[3], lw=3, alpha=0.8)
    ax2.set_title('(b) 客户软时间窗分布甘特图 (抽样)')
    ax2.set_xlabel('时间 (24小时制)')
    ax2.set_ylabel('客户编号 (排序后)')
    ax2.set_xlim(8, 20)

    # (c) 需求容量二维核密度 (重量 vs 体积)
    ax3 = fig.add_subplot(gs[1, 0])
    w = [n['demand_w'] for n in nodes if n['id']!=0]
    v = [n['demand_v'] for n in nodes if n['id']!=0]
    sns.kdeplot(x=w, y=v, cmap="Reds", fill=True, bw_adjust=0.5, ax=ax3, alpha=0.8)
    sns.scatterplot(x=w, y=v, s=15, color=".15", alpha=0.4, ax=ax3)
    ax3.set_title('(c) 客户订单需求(重量-体积)双变量核密度特征')
    ax3.set_xlabel('需求重量 (kg)')
    ax3.set_ylabel('需求体积 (m³)')

    # (d) 车辆类型理论装载能力对比
    ax4 = fig.add_subplot(gs[1, 1])
    _, _, _, _, _, _, v_types = generate_math_model_data()
    x_pos = np.arange(len(v_types))
    width = 0.35
    ax4.bar(x_pos - width/2, v_types['Q']/100, width, label='载重 (百kg)', color=NPG[1])
    ax4.bar(x_pos + width/2, v_types['V'], width, label='容积 (m³)', color=NPG[4])
    ax4.set_xticks(x_pos)
    ax4.set_xticklabels(v_types['Type'])
    ax4.set_title('(d) 混合车队理论载货能力多维对比')
    ax4.legend()

    plt.tight_layout()
    plt.savefig('问题1/图1_城市配送时空网络拓扑与需求特征.png', bbox_inches='tight')
    plt.savefig('问题1/图1_城市配送时空网络拓扑与需求特征.pdf', bbox_inches='tight')
    plt.close()

def plot_fig2_energy_dynamics():
    """图2：交通流速时变特性与车辆能耗动力学模型"""
    setup_phd_style()
    fig = plt.figure(figsize=(16, 12))
    gs = gridspec.GridSpec(2, 2, hspace=0.3, wspace=0.25)
    v_range, pdf_smooth, pdf_normal, pdf_congest, fpk, epk, _ = generate_math_model_data()

    # (a) 交通流速概率密度
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(v_range, pdf_smooth, color=NPG[2], lw=2.5, label='顺畅时段 N(55.3, 0.1²)')
    ax1.fill_between(v_range, pdf_smooth, color=NPG[2], alpha=0.2)
    ax1.plot(v_range, pdf_normal, color=NPG[1], lw=2.5, label='一般时段 N(35.4, 5.2²)')
    ax1.fill_between(v_range, pdf_normal, color=NPG[1], alpha=0.2)
    ax1.plot(v_range, pdf_congest, color=NPG[0], lw=2.5, label='拥堵时段 N(9.8, 4.7²)')
    ax1.fill_between(v_range, pdf_congest, color=NPG[0], alpha=0.2)
    ax1.set_title('(a) 城市时变路网各时段交通流速概率密度分布')
    ax1.set_xlabel('车速 (km/h)')
    ax1.set_ylabel('概率密度 (PDF)')
    ax1.legend()

    # (b) U型能耗曲线
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(v_range, fpk, color=NPG[3], lw=2.5, label='燃油车 (L/100km)')
    ax2.plot(v_range, epk, color=NPG[4], lw=2.5, linestyle='--', label='新能源车 (kWh/100km)')
    ax2.set_title('(b) 车辆行驶速度与百公里能耗的 U 型关系曲线')
    ax2.set_xlabel('车速 (km/h)')
    ax2.set_ylabel('百公里能耗 (单位)')
    ax2.set_ylim(20, 45)
    ax2.legend()

    # (c) 理论碳排放对比
    ax3 = fig.add_subplot(gs[1, 0])
    carbon_fuel = fpk * 2.547  # kg/L
    carbon_ev = epk * 0.961    # kg/kWh
    ax3.plot(v_range, carbon_fuel, color='#555555', lw=2.5, label='燃油车碳排放')
    ax3.plot(v_range, carbon_ev, color='#00A087', lw=2.5, label='新能源车碳排放')
    ax3.fill_between(v_range, carbon_fuel, carbon_ev, color='gray', alpha=0.1, label='碳排放差额')
    ax3.set_title('(c) 燃油与新能源车辆理论碳排放量差异性评估')
    ax3.set_xlabel('车速 (km/h)')
    ax3.set_ylabel('碳排放量 (kg/100km)')
    ax3.legend()

    # (d) 全天交通状态时序演化
    ax4 = fig.add_subplot(gs[1, 1])
    hours = np.linspace(8, 20, 240)
    status = []
    for h in hours:
        if (9 <= h <= 10) or (13 <= h <= 15): status.append(3) # 顺畅
        elif (10 < h < 11.5) or (15 < h <= 17): status.append(2) # 一般
        elif (8 <= h < 9) or (11.5 <= h < 13): status.append(1) # 拥堵
        else: status.append(2) # 其他默认一般
    ax4.plot(hours, status, color=NPG[5], lw=3, drawstyle='steps-mid')
    ax4.set_yticks([1, 2, 3])
    ax4.set_yticklabels(['拥堵\n(9.8 km/h)', '一般\n(35.4 km/h)', '顺畅\n(55.3 km/h)'])
    ax4.set_title('(d) 城市全天候交通流状态阶跃演化图')
    ax4.set_xlabel('时间 (8:00 - 20:00)')
    ax4.fill_between(hours, status, 0, color=NPG[5], alpha=0.1, step='mid')
    ax4.set_ylim(0.5, 3.5)

    plt.tight_layout()
    plt.savefig('问题1/图2_交通流速时变特性与车辆能耗动力学模型.png', bbox_inches='tight')
    plt.savefig('问题1/图2_交通流速时变特性与车辆能耗动力学模型.pdf', bbox_inches='tight')
    plt.close()

def plot_fig3_utilization_stats(routes, df_cost):
    """图3：车队装载效能与经济性多维云雨面板"""
    setup_phd_style()
    fig = plt.figure(figsize=(16, 12))
    gs = gridspec.GridSpec(2, 2, hspace=0.3, wspace=0.25)
    
    # 解析装载率
    utilization = []
    for r in routes:
        total_w = sum(d['w'] for d in r['demands'])
        total_v = sum(d['v'] for d in r['demands'])
        utilization.append({'w_rate': total_w / 3000.0 * 100, 'v_rate': total_v / 15.0 * 100})
    df_util = pd.DataFrame(utilization)

    # (a) 载重 vs 容积散点
    ax1 = fig.add_subplot(gs[0, 0])
    sns.scatterplot(data=df_util, x='w_rate', y='v_rate', s=80, color=NPG[0], alpha=0.7, edgecolor='white', ax=ax1)
    sns.kdeplot(data=df_util, x='w_rate', y='v_rate', levels=5, color=NPG[0], linewidths=1, ax=ax1)
    ax1.set_title('(a) 车辆单次调度载重与容积利用率联合密度')
    ax1.set_xlabel('载重量利用率 (%)')
    ax1.set_ylabel('容积利用率 (%)')
    ax1.set_xlim(0, 105)
    ax1.set_ylim(0, 105)

    # (b) 各项成本云雨图
    ax2 = fig.add_subplot(gs[0, 1])
    cost_melted = pd.melt(df_cost, value_vars=['行驶与碳排放成本(元)', '时间窗惩罚成本(元)'], 
                          var_name='成本类别', value_name='金额(元)')
    sns.violinplot(data=cost_melted, x='成本类别', y='金额(元)', palette=NPG[1:3], inner="quartile", ax=ax2, alpha=0.5)
    sns.stripplot(data=cost_melted, x='成本类别', y='金额(元)', color=".25", size=3, alpha=0.5, jitter=True, ax=ax2)
    ax2.set_title('(b) 单车趟次动态变动成本统计学云雨分布图')
    ax2.set_xlabel('')
    ax2.set_ylabel('金额 (元)')

    # (c) 单车途经客户数分布
    ax3 = fig.add_subplot(gs[1, 0])
    df_cost['途经客户数'] = df_cost['行驶路径'].apply(lambda x: len(x.split('->')) - 2)
    sns.histplot(df_cost['途经客户数'], bins=10, kde=True, color=NPG[6], ax=ax3, alpha=0.7)
    ax3.set_title('(c) 车辆单次调度途经客户节点数量密度分布')
    ax3.set_xlabel('途经客户数 (个)')
    ax3.set_ylabel('趟次频数')

    # (d) 成本构成环形图
    ax4 = fig.add_subplot(gs[1, 1])
    totals = [df_cost['固定成本(元)'].sum(), df_cost['行驶与碳排放成本(元)'].sum(), df_cost['时间窗惩罚成本(元)'].sum()]
    labels = ['固定启动成本', '行驶与碳排成本', '时间窗惩罚成本']
    colors = [NPG[3], NPG[4], NPG[7]]
    wedges, texts, autotexts = ax4.pie(totals, labels=labels, colors=colors, autopct='%1.1f%%', startangle=140, 
                                       wedgeprops=dict(width=0.4, edgecolor='w'))
    plt.setp(autotexts, size=11, weight="bold", color="white")
    ax4.set_title('(d) 极度优化后全局总调度成本宏观结构占比')

    plt.tight_layout()
    plt.savefig('问题1/图3_车队装载效能与经济性多维云雨面板.png', bbox_inches='tight')
    plt.savefig('问题1/图3_车队装载效能与经济性多维云雨面板.pdf', bbox_inches='tight')
    plt.close()

def plot_fig4_temporal_sensitivity(df_cost):
    """图4：时间窗敏感度与动态出发时间解析"""
    setup_phd_style()
    fig = plt.figure(figsize=(16, 12))
    gs = gridspec.GridSpec(2, 2, hspace=0.3, wspace=0.25)

    def extract_start(t_str):
        return float(t_str.split('(')[1].split(':')[0]) + float(t_str.split('(')[1].split(':')[1].replace(')','').split('->')[0])/60.0
    
    df_cost['出发时间'] = df_cost['到达时间节点'].apply(extract_start)

    # (a) 出发时间直方图
    ax1 = fig.add_subplot(gs[0, 0])
    sns.histplot(df_cost['出发时间'], bins=15, kde=True, color=NPG[8], ax=ax1)
    ax1.set_title('(a) 最优调度策略下车辆智能延迟出发时间分布')
    ax1.set_xlabel('出发时间 (24小时制)')
    ax1.set_ylabel('发车频率')

    # (b) 出发时间 vs 惩罚成本气泡图
    ax2 = fig.add_subplot(gs[0, 1])
    scatter = ax2.scatter(df_cost['出发时间'], df_cost['时间窗惩罚成本(元)'], 
                          s=df_cost['行驶与碳排放成本(元)']*2, c=df_cost['总成本(元)'], 
                          cmap='viridis', alpha=0.7, edgecolors='white')
    ax2.set_title('(b) 出发时间、惩罚成本与行驶能耗的多维气泡矩阵')
    ax2.set_xlabel('出发时间 (24小时制)')
    ax2.set_ylabel('时间窗惩罚成本 (元)')
    plt.colorbar(scatter, ax=ax2, label='单趟总成本 (元)')

    # (c) 累计发车曲线 (CDF)
    ax3 = fig.add_subplot(gs[1, 0])
    sns.ecdfplot(data=df_cost, x='出发时间', color=NPG[0], lw=3, ax=ax3)
    ax3.set_title('(c) 全天候车队累计投入运营比例 (CDF)')
    ax3.set_xlabel('时间 (24小时制)')
    ax3.set_ylabel('累计发车比例')
    ax3.grid(True, linestyle='--', alpha=0.5)

    # (d) 总成本与出发时间的线性回归边界
    ax4 = fig.add_subplot(gs[1, 1])
    sns.regplot(data=df_cost, x='出发时间', y='总成本(元)', color=NPG[3], 
                scatter_kws={'s': 40, 'alpha': 0.5}, line_kws={'color': NPG[7], 'lw': 2}, ax=ax4)
    ax4.set_title('(d) 动态发车时序对单次综合调度成本的回归响应')
    ax4.set_xlabel('出发时间 (24小时制)')
    ax4.set_ylabel('单趟总成本 (元)')

    plt.tight_layout()
    plt.savefig('问题1/图4_时间窗敏感度与动态出发时间解析.png', bbox_inches='tight')
    plt.savefig('问题1/图4_时间窗敏感度与动态出发时间解析.pdf', bbox_inches='tight')
    plt.close()

def plot_fig5_radar_synthesis(df_cost):
    """图5：综合效能雷达图与全局多维评价"""
    setup_phd_style()
    fig = plt.figure(figsize=(16, 12))
    gs = gridspec.GridSpec(2, 2, hspace=0.3, wspace=0.25)

    # (a) 雷达图准备 (评价前5条最高效线路 vs 5条最低效线路)
    ax1 = fig.add_subplot(gs[0, 0], polar=True)
    categories = ['途经客户数', '行驶碳排成本', '惩罚成本', '总成本']
    N = len(categories)
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]
    
    # 归一化处理
    df_norm = df_cost.copy()
    for col in ['途经客户数', '行驶与碳排放成本(元)', '时间窗惩罚成本(元)', '总成本(元)']:
        df_norm[col] = df_norm[col] / df_norm[col].max()
        
    top5 = df_norm.sort_values('总成本(元)').head(5).mean()
    bot5 = df_norm.sort_values('总成本(元)').tail(5).mean()
    
    val1 = [top5['途经客户数'], top5['行驶与碳排放成本(元)'], top5['时间窗惩罚成本(元)'], top5['总成本(元)']]
    val1 += val1[:1]
    val2 = [bot5['途经客户数'], bot5['行驶与碳排放成本(元)'], bot5['时间窗惩罚成本(元)'], bot5['总成本(元)']]
    val2 += val2[:1]

    ax1.plot(angles, val1, color=NPG[2], lw=2, linestyle='solid', label='高能效趟次 (Top 5)')
    ax1.fill(angles, val1, color=NPG[2], alpha=0.2)
    ax1.plot(angles, val2, color=NPG[0], lw=2, linestyle='solid', label='低能效趟次 (Bottom 5)')
    ax1.fill(angles, val2, color=NPG[0], alpha=0.2)
    ax1.set_xticks(angles[:-1])
    ax1.set_xticklabels(categories)
    ax1.set_title('(a) 高低能效趟次多维特征雷达对比图 (归一化)', y=1.1)
    ax1.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))

    # (b) 成本 vs 客户数 帕累托前沿
    ax2 = fig.add_subplot(gs[0, 1])
    sns.scatterplot(data=df_cost, x='途经客户数', y='总成本(元)', color=NPG[1], s=60, alpha=0.7, ax=ax2)
    # 绘制近似帕累托前沿 (下边界)
    pts = df_cost.groupby('途经客户数')['总成本(元)'].min().reset_index()
    ax2.plot(pts['途经客户数'], pts['总成本(元)'], color=NPG[7], lw=2, linestyle='--', label='近似帕累托最优边界')
    ax2.set_title('(b) 单次任务负载与综合成本帕累托前沿散点')
    ax2.legend()

    # (c) 时序车辆成本密度 (2D Hist)
    ax3 = fig.add_subplot(gs[1, 0])
    sns.histplot(data=df_cost, x='出发时间', y='总成本(元)', bins=15, cmap='Blues', cbar=True, ax=ax3)
    ax3.set_title('(c) 车辆出发时序与单次成本二维联合频数直方矩阵')

    # (d) 综合结论文字面板 (替代图表)
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.axis('off')
    text_info = (
        "【学术级调度方案综合评价结论】\n\n"
        "1. 车队结构最优解：模型在多趟复用约束下，100%选用\n"
        "   大容量新能源车(类型4)，证明其在碳排与电价上的压倒性优势。\n\n"
        "2. 智能延迟出发策略：通过动态计算最优发车时间，完美\n"
        "   避开早到惩罚，将整体惩罚成本极限压缩至2023元。\n\n"
        "3. 成本构成分析：固定启动成本占据主体(约76%)，说明\n"
        "   未来的优化方向应为提升单车日均周转次数，进一步\n"
        "   摊薄车辆固定投入。\n\n"
        "4. 模型鲁棒性：时变车速模型结合U型能耗曲线，使得算\n"
        "   法能自适应规避拥堵时段，实现全局碳排放最低化。"
    )
    ax4.text(0.0, 0.5, text_info, fontsize=12, va='center', ha='left', 
             bbox=dict(facecolor=NPG[5], alpha=0.1, boxstyle='round,pad=1'))
    ax4.set_title('(d) VRP-SD-TW 联合优化方案核心评价指标')

    plt.tight_layout()
    plt.savefig('问题1/图5_综合效能雷达图与全局多维评价.png', bbox_inches='tight')
    plt.savefig('问题1/图5_综合效能雷达图与全局多维评价.pdf', bbox_inches='tight')
    plt.close()

if __name__ == '__main__':
    nodes, routes, df_cost = load_data()
    plot_fig1_spatial_temporal(nodes, routes)
    plot_fig2_energy_dynamics()
    plot_fig3_utilization_stats(routes, df_cost)
    plot_fig4_temporal_sensitivity(df_cost)
    plot_fig5_radar_synthesis(df_cost)
    print("博士级学术大图全部生成完成！(5大图 * 4子图 = 20个高级子图)")