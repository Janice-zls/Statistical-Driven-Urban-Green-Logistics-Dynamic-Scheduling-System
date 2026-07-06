import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.patches as patches
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset

def setup_chinese_sci_style():
    """设置学术级全中文无衬线绘图风格"""
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
        'axes.titlesize': 16,
        'axes.titleweight': 'bold',
        'axes.labelsize': 14,
        'legend.fontsize': 11,
        'legend.frameon': False,
        'figure.dpi': 300
    })

# Nature NPG Palette
NPG_COLORS = ['#E64B35', '#4DBBD5', '#00A087', '#3C5488', '#F39B7F', '#8491B4', '#91D1C2', '#DC0000']

VEHICLE_MAP = {
    1: {'name': '燃油车(3.0t)', 'color': NPG_COLORS[0], 'Q': 3000, 'V': 13.5},
    2: {'name': '燃油车(1.5t)', 'color': NPG_COLORS[4], 'Q': 1500, 'V': 10.8},
    3: {'name': '燃油车(1.25t)', 'color': NPG_COLORS[3], 'Q': 1250, 'V': 6.5},
    4: {'name': '新能源车(3.0t)', 'color': NPG_COLORS[2], 'Q': 3000, 'V': 15.0},
    5: {'name': '新能源车(1.25t)', 'color': NPG_COLORS[1], 'Q': 1250, 'V': 8.5}
}

def load_data():
    with open('问题1/processed_data.json', 'r', encoding='utf-8') as f:
        data = json.load(f)
    with open('问题1/solution_opt.json', 'r', encoding='utf-8') as f:
        sol = json.load(f)
    df_cost = pd.read_excel('问题1/车辆调度方案.xlsx')
    return data['nodes'], sol['routes'], df_cost

def plot_1_spatial_map(nodes, routes):
    """图1：城市物流配送路线空间分布图（带核心区局部放大）"""
    fig, ax = plt.subplots(figsize=(10, 10))
    
    # 绘制绿色配送区
    gz = patches.Circle((0, 0), 10, facecolor=NPG_COLORS[2], alpha=0.08, 
                        edgecolor=NPG_COLORS[2], linestyle='--', linewidth=2, zorder=1, label='绿色配送区 (r=10km)')
    ax.add_patch(gz)
    
    # 绘制路径
    plotted_v = set()
    for r in routes:
        v_id = r['vehicle']
        v_info = VEHICLE_MAP[v_id]
        route_nodes = r['route']
        x = [nodes[n]['x'] for n in route_nodes]
        y = [nodes[n]['y'] for n in route_nodes]
        
        lbl = v_info['name'] if v_id not in plotted_v else ""
        ax.plot(x, y, color=v_info['color'], alpha=0.6, linewidth=1.5, zorder=2, label=lbl)
        plotted_v.add(v_id)
        
    # 绘制客户点与配送中心
    cx = [n['x'] for n in nodes if n['id'] != 0]
    cy = [n['y'] for n in nodes if n['id'] != 0]
    ax.scatter(cx, cy, s=30, c=NPG_COLORS[5], alpha=0.7, edgecolors='white', linewidth=0.5, zorder=3, label='客户节点')
    
    dx, dy = nodes[0]['x'], nodes[0]['y']
    ax.scatter([dx], [dy], s=350, c=NPG_COLORS[7], marker='*', edgecolors='black', linewidths=1, zorder=4, label='配送中心 (Depot)')
    
    ax.set_xlabel('X坐标 (km)')
    ax.set_ylabel('Y坐标 (km)')
    ax.set_title('图1：城市物流配送调度路线空间分布图', loc='left')
    ax.grid(True, linestyle=':', alpha=0.6)
    ax.set_aspect('equal', adjustable='datalim')
    
    # 重组图例
    handles, labels = ax.get_legend_handles_labels()
    # 过滤空label
    h_l = [(h, l) for h, l in zip(handles, labels) if l]
    ax.legend([h for h, l in h_l], [l for h, l in h_l], loc='upper left', bbox_to_anchor=(1.02, 1))
    
    # 局部放大（聚焦市中心）
    axins = inset_axes(ax, width="40%", height="40%", loc='lower left', 
                       bbox_to_anchor=(0.03, 0.03, 1, 1), bbox_transform=ax.transAxes)
    axins.add_patch(patches.Circle((0, 0), 10, facecolor=NPG_COLORS[2], alpha=0.1, 
                                   edgecolor=NPG_COLORS[2], linestyle='--', linewidth=1.5, zorder=1))
    
    for r in routes:
        v_info = VEHICLE_MAP[r['vehicle']]
        route_nodes = r['route']
        x = [nodes[n]['x'] for n in route_nodes]
        y = [nodes[n]['y'] for n in route_nodes]
        axins.plot(x, y, color=v_info['color'], alpha=0.7, linewidth=1.2, zorder=2)
        
    axins.scatter(cx, cy, s=40, c=NPG_COLORS[5], alpha=0.9, edgecolors='white', zorder=3)
    axins.scatter([0], [0], s=120, c='black', marker='P', zorder=4, label='市中心')
    
    axins.set_xlim(-12, 12)
    axins.set_ylim(-12, 12)
    axins.set_xticklabels([])
    axins.set_yticklabels([])
    axins.tick_params(bottom=False, left=False)
    for spine in axins.spines.values():
        spine.set_edgecolor('gray')
        spine.set_linewidth(1.5)
        spine.set_visible(True)
        
    mark_inset(ax, axins, loc1=2, loc2=4, fc="none", ec="0.5", alpha=0.7, linestyle='--')
    
    plt.savefig('问题1/图1_城市物流配送路线空间分布图.png', bbox_inches='tight')
    plt.savefig('问题1/图1_城市物流配送路线空间分布图.pdf', bbox_inches='tight')
    plt.close()

def plot_2_cost_composition(df_cost):
    """图2：各类型车辆成本构成分析（堆叠柱状图）"""
    df_cost['车辆类型名称'] = df_cost['车辆类型'].map(lambda x: VEHICLE_MAP[x]['name'])
    
    # 汇总各类型车辆的各项成本
    cost_sum = df_cost.groupby('车辆类型名称')[['固定成本(元)', '行驶与碳排放成本(元)', '时间窗惩罚成本(元)']].sum()
    
    # 按照总成本排序
    cost_sum['总成本'] = cost_sum.sum(axis=1)
    cost_sum = cost_sum.sort_values('总成本', ascending=False).drop(columns=['总成本'])
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    cost_sum.plot(kind='bar', stacked=True, ax=ax, 
                  color=[NPG_COLORS[5], NPG_COLORS[1], NPG_COLORS[0]], alpha=0.85, width=0.6)
                  
    ax.set_xlabel('调用的车辆类型')
    ax.set_ylabel('累计成本总额 (元)')
    ax.set_title('图2：各类型车辆整体配送成本构成分析', loc='left')
    plt.xticks(rotation=0)
    ax.grid(axis='y', linestyle='--', alpha=0.4)
    ax.legend(title='成本类别', loc='upper right')
    
    # 添加数值标签
    for c in ax.containers:
        ax.bar_label(c, label_type='center', fmt='%.0f', color='white', fontweight='bold', fontsize=10)
        
    plt.savefig('问题1/图2_各类型车辆成本构成分析图.png', bbox_inches='tight')
    plt.savefig('问题1/图2_各类型车辆成本构成分析图.pdf', bbox_inches='tight')
    plt.close()

def plot_3_raincloud_distribution(df_cost):
    """图3：单车配送成本云雨分布图 (Raincloud / Violin Plot)"""
    df_cost['车辆类型名称'] = df_cost['车辆类型'].map(lambda x: VEHICLE_MAP[x]['name'])
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    order = [VEHICLE_MAP[i]['name'] for i in [4, 1, 5, 2, 3] if VEHICLE_MAP[i]['name'] in df_cost['车辆类型名称'].values]
    
    # 小提琴图
    sns.violinplot(data=df_cost, x='车辆类型名称', y='总成本(元)', order=order,
                   palette=[VEHICLE_MAP[i]['color'] for i in [4, 1, 5, 2, 3] if i in VEHICLE_MAP], 
                   inner="box", alpha=0.5, ax=ax, linewidth=1.5)
                   
    # 散点抖动图
    sns.stripplot(data=df_cost, x='车辆类型名称', y='总成本(元)', order=order,
                  color="black", alpha=0.4, jitter=True, size=5, ax=ax)
                  
    ax.set_xlabel('车辆类型')
    ax.set_ylabel('单车总配送成本 (元)')
    ax.set_title('图3：各类型车辆单趟配送成本的核密度与散点分布图 (Raincloud Plot)', loc='left')
    ax.grid(axis='y', linestyle='--', alpha=0.4)
    
    plt.savefig('问题1/图3_各类型车辆单趟配送成本分布图.png', bbox_inches='tight')
    plt.savefig('问题1/图3_各类型车辆单趟配送成本分布图.pdf', bbox_inches='tight')
    plt.close()

def plot_4_capacity_utilization(nodes, routes):
    utilization = []
    for r in routes:
        v_id = r['vehicle']
        v_info = VEHICLE_MAP[v_id]
        
        # Calculate actual loaded weight and volume from the route demands
        total_w = sum(d['w'] for d in r['demands'])
        total_v = sum(d['v'] for d in r['demands'])
        
        w_rate = total_w / v_info['Q'] * 100
        v_rate = total_v / v_info['V'] * 100
        
        utilization.append({
            '车辆类型': v_info['name'],
            '载重利用率(%)': w_rate,
            '容积利用率(%)': v_rate,
            '颜色': v_info['color']
        })
        
    df_util = pd.DataFrame(utilization)
    
    fig, ax = plt.subplots(figsize=(9, 7))
    
    for v_name, group in df_util.groupby('车辆类型'):
        color = group['颜色'].iloc[0]
        ax.scatter(group['载重利用率(%)'], group['容积利用率(%)'], 
                   c=color, s=80, alpha=0.7, edgecolors='white', label=v_name)
                   
    ax.axhline(100, color='gray', linestyle='--', alpha=0.5)
    ax.axvline(100, color='gray', linestyle='--', alpha=0.5)
    
    # 标注理想区间
    ax.fill_between([80, 100], 80, 100, color=NPG_COLORS[6], alpha=0.1, label='高利用率区间 (80%-100%)')
    
    ax.set_xlabel('车辆载重利用率 (%)')
    ax.set_ylabel('车辆容积利用率 (%)')
    ax.set_title('图4：车辆载重与容积双维利用率分析散点图', loc='left')
    ax.grid(True, linestyle=':', alpha=0.5)
    ax.legend(loc='lower left')
    
    plt.savefig('问题1/图4_车辆装载率与容积利用率分析图.png', bbox_inches='tight')
    plt.savefig('问题1/图4_车辆装载率与容积利用率分析图.pdf', bbox_inches='tight')
    plt.close()

def generate_all_plots():
    setup_chinese_sci_style()
    nodes, routes, df_cost = load_data()
    
    plot_1_spatial_map(nodes, routes)
    plot_2_cost_composition(df_cost)
    plot_3_raincloud_distribution(df_cost)
    plot_4_capacity_utilization(nodes, routes)
    print("4张高质量学术配图已全部生成。")

if __name__ == '__main__':
    generate_all_plots()