import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.gridspec import GridSpec
import warnings
warnings.filterwarnings('ignore')

# 从博士级主算法导入核心模型和环境
from phd_alns_dynamic_vrptw import DynamicVRPTW, AdvancedSolver, simulate_dynamic_events, START_COST, VEHICLE_TYPES

# ---------------------------------------------------------
# 全局美学设置 (顶级学术期刊 SCI/Nature 风格)
# ---------------------------------------------------------
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['axes.linewidth'] = 1.5
plt.rcParams['xtick.major.width'] = 1.5
plt.rcParams['ytick.major.width'] = 1.5
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300

# NPG Palette
NPG_RED = '#E64B35'
NPG_CYAN = '#4DBBD5'
NPG_GREEN = '#00A087'
NPG_BLUE = '#3C5488'
NPG_ORANGE = '#F39B7F'
NPG_GREY = '#8491B4'

def get_detailed_metrics():
    print("Running Core Model to extract metrics...")
    env = DynamicVRPTW(r"../附件")
    solver = AdvancedSolver(env)
    
    # 抑制仿真打印
    sys.stdout = open(os.devnull, 'w')
    s_routes, s_vtypes = solver.build_initial_routes(range(1, env.num_nodes))
    s_cost, s_details = solver.evaluate_routes(s_routes, s_vtypes)
    d_routes, d_vtypes = simulate_dynamic_events(env, solver, s_routes, s_vtypes)
    d_cost, d_details = solver.evaluate_routes(d_routes, d_vtypes)
    sys.stdout = sys.__stdout__
    
    # 构建结构化 DataFrame
    df_s = pd.DataFrame(s_details)
    df_s['Scenario'] = '静态基线调度'
    
    df_d = pd.DataFrame(d_details)
    df_d['Scenario'] = '动态事件重调度'
    
    df = pd.concat([df_s, df_d], ignore_index=True)
    df['actual_tw_cost'] = df['tw_cost'] - START_COST
    df['start_cost'] = START_COST
    df['v_cat'] = df['vtype'].apply(lambda x: '新能源车 (EV)' if VEHICLE_TYPES[x]['type']=='EV' else '燃油车 (Fuel)')
    
    return df, s_cost, d_cost

def plot_economic_environmental_dashboard(df):
    """ 图1：经济与环保多维效益分析面板 (2x2) """
    print("Generating Dashboard 1 (2x2)...")
    fig = plt.figure(figsize=(16, 12), dpi=300)
    fig.suptitle('经济与环保多维效益深度分析 (Economic & Environmental Dashboard)', fontsize=24, fontweight='bold', y=0.98)
    
    gs = GridSpec(2, 2, figure=fig, wspace=0.25, hspace=0.3)
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[1, 0])
    ax4 = fig.add_subplot(gs[1, 1])
    
    # ==============================
    # (a) 运行效能多维气泡状态空间 (Bubble Scatter)
    # ==============================
    sns.scatterplot(data=df, x='dist', y='total', size='carbon_cost', sizes=(50, 600), 
                    hue='v_cat', palette={"燃油车 (Fuel)": NPG_RED, "新能源车 (EV)": NPG_GREEN}, 
                    alpha=0.75, edgecolor='black', ax=ax1)
    ax1.set_title('(a) 车辆运行效能多维状态空间 (Bubble Space)', fontsize=16, fontweight='bold', pad=10)
    ax1.set_xlabel('单车行驶距离 (km)', fontsize=14)
    ax1.set_ylabel('单车综合总成本 (元)', fontsize=14)
    
    # 调整图例避免过长
    handles, labels = ax1.get_legend_handles_labels()
    ax1.legend(handles=handles[:6], labels=labels[:6], fontsize=10, framealpha=0.9, edgecolor='black', loc='best', ncol=2)
    ax1.grid(True, linestyle='--', alpha=0.5)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)

    # ==============================
    # (b) 动态重调度成本构成 (Donut Chart)
    # ==============================
    dyn_df = df[df['Scenario'] == '动态事件重调度']
    cost_sums = [
        dyn_df['start_cost'].sum(),
        dyn_df['energy_cost'].sum(),
        dyn_df['actual_tw_cost'].sum(),
        dyn_df['carbon_cost'].sum()
    ]
    labels = ['车辆启动成本\n(Start Cost)', '能耗成本\n(Energy Cost)', '时间窗惩罚\n(Penalty)', '碳排放成本\n(Carbon)']
    explode = (0.05, 0.05, 0.15, 0.1) # 突出惩罚和碳排
    pie_colors = [NPG_GREY, NPG_BLUE, NPG_RED, NPG_GREEN]
    
    wedges, texts, autotexts = ax2.pie(cost_sums, explode=explode, labels=labels, colors=pie_colors, 
                                       autopct='%1.1f%%', shadow=True, startangle=140, 
                                       textprops=dict(color="black", fontsize=12, fontweight='bold'))
    # Draw circle for Donut shape
    centre_circle = plt.Circle((0,0), 0.55, fc='white')
    ax2.add_artist(centre_circle)
    ax2.set_title('(b) 动态调度系统总成本构成', fontsize=16, fontweight='bold', pad=10)

    # ==============================
    # (c) 距离与碳排双变量核密度 (Bivariate KDE)
    # ==============================
    sns.kdeplot(data=df, x='dist', y='carbon_cost', hue='v_cat', fill=True, 
                alpha=0.5, thresh=0.05, levels=8, palette={"燃油车 (Fuel)": NPG_RED, "新能源车 (EV)": NPG_GREEN}, ax=ax3)
    sns.scatterplot(data=df, x='dist', y='carbon_cost', hue='v_cat', 
                    s=25, alpha=0.9, palette={"燃油车 (Fuel)": NPG_RED, "新能源车 (EV)": NPG_GREEN}, edgecolor='white', ax=ax3, legend=False)
    
    ax3.set_title('(c) 距离-碳排双变量联合概率密度 (Bivariate KDE)', fontsize=16, fontweight='bold', pad=10)
    ax3.set_xlabel('单车行驶距离 (km)', fontsize=14)
    ax3.set_ylabel('单车碳排放成本 (元)', fontsize=14)
    ax3.legend(title='动力类型', fontsize=11, framealpha=0.9, edgecolor='black', loc='upper left')
    ax3.grid(True, linestyle='--', alpha=0.5)
    ax3.spines['top'].set_visible(False)
    ax3.spines['right'].set_visible(False)

    # ==============================
    # (d) 能耗成本与时间窗惩罚的帕累托关系 (Scatter)
    # ==============================
    sns.scatterplot(x='energy_cost', y='actual_tw_cost', hue='v_cat', style='Scenario', 
                    data=df, s=150, alpha=0.8, palette=[NPG_RED, NPG_GREEN], edgecolor='black', ax=ax4)
    
    ax4.set_title('(d) 单车能耗与时间窗惩罚权衡关系', fontsize=16, fontweight='bold', pad=10)
    ax4.set_xlabel('单车能耗成本 (元)', fontsize=14)
    ax4.set_ylabel('时间窗违规惩罚成本 (元)', fontsize=14)
    ax4.legend(fontsize=11, framealpha=0.9, edgecolor='black', loc='upper right')
    ax4.grid(True, linestyle=':', alpha=0.6)
    ax4.spines['top'].set_visible(False)
    ax4.spines['right'].set_visible(False)
    
    plt.savefig('多维子图_1_经济环保分析_SCI.png', bbox_inches='tight')

def plot_operational_dashboard(df):
    """ 图2：异构车队运行特征深度剖析面板 (1x3) """
    print("Generating Dashboard 2 (1x3)...")
    fig, axes = plt.subplots(1, 3, figsize=(20, 6), dpi=300)
    fig.suptitle('异构车队微观运行特征剖析 (Operational Efficiency Dashboard)', fontsize=22, fontweight='bold', y=1.05)
    
    # ==============================
    # (a) 各车型行驶距离箱线+蜂群图
    # ==============================
    ax1 = axes[0]
    sns.boxplot(x='vtype', y='dist', data=df[df['Scenario'] == '动态事件重调度'], 
                ax=ax1, palette="Set2", showfliers=False, width=0.5, boxprops=dict(alpha=0.6))
    sns.swarmplot(x='vtype', y='dist', data=df[df['Scenario'] == '动态事件重调度'], 
                  ax=ax1, color=".25", size=6, alpha=0.8)
                  
    ax1.set_title('(a) 各车型单车行驶距离分布', fontsize=15, fontweight='bold', pad=10)
    ax1.set_ylabel('行驶距离 (km)', fontsize=13)
    ax1.set_xlabel('车型', fontsize=13)
    ax1.grid(axis='y', linestyle='--', alpha=0.5)
    
    # ==============================
    # (b) 燃油车 vs 新能源车 成本雷达对比 (改为水平条形图展示绝对差异)
    # ==============================
    ax2 = axes[1]
    agg_df = df[df['Scenario'] == '动态事件重调度'].groupby('v_cat')[['energy_cost', 'carbon_cost', 'actual_tw_cost']].mean()
    agg_df.columns = ['平均能耗', '平均碳排', '平均时间惩罚']
    
    agg_df.T.plot(kind='barh', ax=ax2, color=[NPG_RED, NPG_GREEN], edgecolor='black', alpha=0.85)
    ax2.set_title('(b) 油/电车型平均作业代价对比', fontsize=15, fontweight='bold', pad=10)
    ax2.set_xlabel('平均成本 (元)', fontsize=13)
    ax2.legend(fontsize=11, framealpha=0.9, edgecolor='black')
    ax2.grid(axis='x', linestyle='--', alpha=0.5)
    
    # ==============================
    # (c) 动态事件造成的性能偏移 (KDE 分布偏移)
    # ==============================
    ax3 = axes[2]
    sns.kdeplot(data=df, x='total', hue='Scenario', fill=True, common_norm=False, 
                palette=[NPG_BLUE, NPG_ORANGE], alpha=0.5, linewidth=2, ax=ax3)
                
    ax3.set_title('(c) 系统单车总成本分布偏移', fontsize=15, fontweight='bold', pad=10)
    ax3.set_xlabel('单车综合总成本 (元)', fontsize=13)
    ax3.set_ylabel('核密度 (Density)', fontsize=13)
    ax3.grid(True, linestyle=':', alpha=0.6)
    
    for ax in axes:
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.tick_params(labelsize=12)

    plt.tight_layout()
    plt.savefig('多维子图_2_运行特征剖析_SCI.png', bbox_inches='tight')

if __name__ == "__main__":
    df, s_cost, d_cost = get_detailed_metrics()
    plot_economic_environmental_dashboard(df)
    plot_operational_dashboard(df)
    print("All stunning visualizations generated successfully!")
