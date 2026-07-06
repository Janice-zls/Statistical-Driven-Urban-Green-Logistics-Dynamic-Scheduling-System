import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import matplotlib.gridspec as gridspec
import seaborn as sns
import warnings
import matplotlib.patheffects as path_effects

warnings.filterwarnings('ignore')

def setup_sci_nature_style():
    """纯正的 Nature/Science 顶级期刊极简学术风格"""
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['SimHei', 'Microsoft YaHei', 'Arial'],
        'axes.unicode_minus': False,
        'axes.spines.top': False,
        'axes.spines.right': False,
        'axes.linewidth': 1.2,
        'xtick.major.width': 1.2,
        'ytick.major.width': 1.2,
        'xtick.direction': 'out',
        'ytick.direction': 'out',
        'axes.grid': False,
        'axes.facecolor': 'white',
        'figure.facecolor': 'white',
        'font.size': 12,
        'axes.titlesize': 14,
        'axes.titleweight': 'bold',
        'axes.labelsize': 12,
        'legend.frameon': False,
        'legend.fontsize': 11,
        'figure.dpi': 300
    })

NPG_COLORS = ['#E64B35', '#4DBBD5', '#00A087', '#3C5488', '#F39B7F', '#8491B4', '#91D1C2', '#DC0000']

def calc_carbon(row):
    cost = row['行驶与碳排放成本(元)']
    v_type = row['车辆类型']
    if v_type in [1, 2, 3]: # Fuel
        return cost * (2.547 / 9.26555)
    else: # Elec
        return cost * (0.961 / 2.26465)

def analyze_and_plot():
    setup_sci_nature_style()
    
    df1 = pd.read_excel('问题1/车辆调度方案.xlsx')
    df2 = pd.read_excel('问题2/车辆调度方案.xlsx')
    
    df1['碳排放量(kg)'] = df1.apply(calc_carbon, axis=1)
    df2['碳排放量(kg)'] = df2.apply(calc_carbon, axis=1)
    
    # 汇总数据
    c1 = df1['总成本(元)'].sum()
    c2 = df2['总成本(元)'].sum()
    
    cb1 = df1['碳排放量(kg)'].sum()
    cb2 = df2['碳排放量(kg)'].sum()
    
    # 车辆使用结构
    v_counts1 = df1['车辆类型'].value_counts().sort_index()
    v_counts2 = df2['车辆类型'].value_counts().sort_index()
    
    all_v_types = sorted(list(set(v_counts1.keys()).union(set(v_counts2.keys()))))
    c_v1 = [v_counts1.get(k, 0) for k in all_v_types]
    c_v2 = [v_counts2.get(k, 0) for k in all_v_types]
    
    fig = plt.figure(figsize=(18, 12))
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.3, wspace=0.2)
    fig.suptitle("问题2：环保限行政策 (绿色配送区) 影响深度剖析", fontsize=22, fontweight='bold', y=0.95)

    # (a) 成本与碳排放双轴对比柱状图 (Dual-axis Bar)
    ax1 = fig.add_subplot(gs[0, 0])
    x = np.arange(2)
    width = 0.35
    
    costs = [c1, c2]
    carbons = [cb1, cb2]
    
    bar1 = ax1.bar(x - width/2, costs, width, color=NPG_COLORS[3], label='总成本 (元)', edgecolor='black', lw=1.2)
    ax1.set_ylabel('总调度成本 (元)', color=NPG_COLORS[3], fontweight='bold')
    ax1.tick_params(axis='y', labelcolor=NPG_COLORS[3])
    ax1.set_xticks(x)
    ax1.set_xticklabels(['无政策 (问题1)', '限行政策 (问题2)'], fontweight='bold')
    
    # 添加成本标注
    for i, v in enumerate(costs):
        ax1.text(x[i] - width/2, v + max(costs)*0.02, f"¥{v:,.0f}", ha='center', va='bottom', 
                 color=NPG_COLORS[3], fontweight='bold', fontsize=11)
    
    ax1_2 = ax1.twinx()
    bar2 = ax1_2.bar(x + width/2, carbons, width, color=NPG_COLORS[0], label='总碳排放 (kg)', edgecolor='black', lw=1.2)
    ax1_2.set_ylabel('总碳排放量 (kg)', color=NPG_COLORS[0], fontweight='bold')
    ax1_2.tick_params(axis='y', labelcolor=NPG_COLORS[0])
    ax1_2.spines['top'].set_visible(False)
    
    # 添加碳排放标注
    for i, v in enumerate(carbons):
        ax1_2.text(x[i] + width/2, v + max(carbons)*0.02, f"{v:,.0f} kg", ha='center', va='bottom', 
                   color=NPG_COLORS[0], fontweight='bold', fontsize=11)
    
    ax1.set_title('(a) 环保政策对调度总成本与碳排放的宏观影响', pad=15)
    
    # (b) 车辆使用结构演变 (Slope Chart / Dumbbell Plot)
    ax2 = fig.add_subplot(gs[0, 1])
    
    v_labels = {
        1: "燃油车(3t)", 
        2: "燃油车(1.5t)", 
        3: "燃油车(1.25t)", 
        4: "新能源(3t)", 
        5: "新能源(1.25t)"
    }
    
    ax2.axvline(0, color='gray', linestyle='-', lw=1, alpha=0.5)
    ax2.axvline(1, color='gray', linestyle='-', lw=1, alpha=0.5)
    
    for i, v_type in enumerate(all_v_types):
        y1 = c_v1[i]
        y2 = c_v2[i]
        color = NPG_COLORS[2] if v_type in [4, 5] else NPG_COLORS[1] # 绿表示新能源，蓝表示燃油
        
        ax2.plot([0, 1], [y1, y2], marker='o', markersize=10, color=color, lw=3, label=v_labels[v_type])
        
        # 标注文字
        ax2.text(-0.05, y1, f"{v_labels[v_type]}: {y1}辆", ha='right', va='center', fontsize=11, fontweight='bold')
        ax2.text(1.05, y2, f"{y2}辆", ha='left', va='center', fontsize=11, fontweight='bold')
        
    ax2.set_xticks([0, 1])
    ax2.set_xticklabels(['无政策 (问题1)', '限行政策 (问题2)'], fontweight='bold', fontsize=12)
    ax2.set_yticks([])
    ax2.spines['left'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.spines['bottom'].set_visible(False)
    ax2.set_xlim(-0.5, 1.5)
    ax2.set_title('(b) 车辆调度结构的战略转移 (Slope Chart)', pad=15)
    
    # 添加图例
    handles, labels = ax2.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax2.legend(by_label.values(), by_label.keys(), loc='upper center', bbox_to_anchor=(0.5, 0.95), ncol=2)

    # (c) 单车碳排放密度分布差异 (KDE Plot)
    ax3 = fig.add_subplot(gs[1, 0])
    sns.kdeplot(df1['碳排放量(kg)'], ax=ax3, color=NPG_COLORS[3], fill=True, alpha=0.3, lw=2, label='无政策 (问题1)')
    sns.kdeplot(df2['碳排放量(kg)'], ax=ax3, color=NPG_COLORS[0], fill=True, alpha=0.3, lw=2, label='限行政策 (问题2)')
    ax3.set_xlabel('单车航次碳排放量 (kg)')
    ax3.set_ylabel('概率密度 (Density)')
    ax3.set_title('(c) 单车航次碳排放密度的分布变迁 (KDE)', pad=15)
    ax3.legend()

    # (d) 绿色配送区限行约束下的路径时间窗偏移分析 (Scatter Plot)
    ax4 = fig.add_subplot(gs[1, 1])
    
    # 提取到达时间 (以小时计)
    def extract_hours(df):
        times = []
        for _, row in df.iterrows():
            time_str = row['到达时间节点']
            # 形如 "0(08:00) -> 15(08:15) -> 0(09:30)"
            nodes = time_str.split(' -> ')
            for n in nodes:
                try:
                    nid = int(n.split('(')[0])
                    if nid != 0: # 仅看客户点
                        t_val = n.split('(')[1].split(')')[0]
                        h, m = map(int, t_val.split(':'))
                        times.append(h + m/60.0)
                except:
                    pass
        return times
        
    t1 = extract_hours(df1)
    t2 = extract_hours(df2)
    
    # 绘制直方图分布
    bins = np.linspace(8, 24, 32)
    ax4.hist(t1, bins=bins, alpha=0.5, color=NPG_COLORS[1], label='无政策', edgecolor='white', density=True)
    ax4.hist(t2, bins=bins, alpha=0.5, color=NPG_COLORS[2], label='限行政策', edgecolor='white', density=True)
    
    # 标出禁行时间段 8:00 - 16:00
    ax4.axvspan(8, 16, color=NPG_COLORS[0], alpha=0.1, label='燃油车限行时段 (8:00-16:00)')
    
    ax4.set_xlabel('到达客户点的时间分布 (Hour)')
    ax4.set_ylabel('配送频次密度')
    ax4.set_title('(d) 客户服务时间重构与限行规避效应分析', pad=15)
    ax4.set_xlim(8, 24)
    ax4.set_xticks([8, 12, 16, 20, 24])
    ax4.set_xticklabels(['08:00', '12:00', '16:00', '20:00', '24:00'])
    ax4.legend(loc='upper right')

    plt.savefig('问题2/政策影响分析对比大图.png', bbox_inches='tight')
    plt.savefig('问题2/政策影响分析对比大图.pdf', bbox_inches='tight')
    plt.close()
    
if __name__ == '__main__':
    analyze_and_plot()
    print("【真·SCI极简风】政策影响分析对比大图生成完毕！")
