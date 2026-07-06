import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# 引入核心算法类
from ultimate_sci_visualize import UltimateDynamicVRP

# ---------------------------------------------------------
# SCI/Nature 顶级排版与中文字体设置
# ---------------------------------------------------------
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS'] # 中文字体支持
plt.rcParams['axes.unicode_minus'] = False # 负号正常显示
plt.rcParams['axes.linewidth'] = 1.5
plt.rcParams['xtick.major.width'] = 1.5
plt.rcParams['ytick.major.width'] = 1.5
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300

# Nature 经典高对比度配色 (NPG Palette)
NPG_COLORS = ['#3C5488', '#E64B35', '#00A087', '#4DBBD5']

def calc_metrics(env, routes):
    distances = []
    loads = []
    for r in routes:
        dist = 0
        load = 0
        for i in range(len(r)-1):
            dist += env.dist_matrix[r[i], r[i+1]]
        for node in r:
            if node != 0 and node < len(env.demands):
                load += env.demands[node, 0]
        distances.append(dist)
        loads.append(load)
    return distances, loads

def main():
    print("Running Model and Extracting Metrics...")
    env = UltimateDynamicVRP(r"../附件")
    static_routes = env.generate_initial_solution()
    dynamic_routes = env.process_dynamic_events(static_routes)
    
    static_dist, static_loads = calc_metrics(env, static_routes)
    dyn_dist, dyn_loads = calc_metrics(env, dynamic_routes)
    
    # =========================================================
    # 图2: 宏观性能指标对比 (SCI 柱状图)
    # =========================================================
    print("Generating Figure 2: Macro Metrics Bar Chart...")
    fig, axes = plt.subplots(1, 3, figsize=(16, 5), dpi=300)
    fig.suptitle('动态重调度宏观性能指标评估 (Macro Performance Metrics)', fontsize=18, fontweight='bold', y=1.05)
    
    metrics = [
        ('系统总行驶距离 (km)', sum(static_dist), sum(dyn_dist)),
        ('单车平均行驶距离 (km)', np.mean(static_dist), np.mean(dyn_dist)),
        ('调度车辆总数 (辆)', len(static_routes), len(dynamic_routes))
    ]
    
    for i, (title, v_stat, v_dyn) in enumerate(metrics):
        bars = axes[i].bar(['静态基线\n(Static)', '动态重调度\n(Dynamic)'], [v_stat, v_dyn], 
                           color=[NPG_COLORS[0], NPG_COLORS[1]], width=0.5, edgecolor='black', linewidth=1.5, alpha=0.9)
        axes[i].set_title(title, fontsize=15, pad=15)
        axes[i].spines['top'].set_visible(False)
        axes[i].spines['right'].set_visible(False)
        axes[i].grid(axis='y', linestyle='--', alpha=0.5)
        
        # 添加数据标签
        for bar, val in zip(bars, [v_stat, v_dyn]):
            axes[i].text(bar.get_x() + bar.get_width()/2, val + (max(v_stat, v_dyn) * 0.02), 
                         f"{val:.2f}" if isinstance(val, float) else f"{val}", 
                         ha='center', va='bottom', fontsize=13, fontweight='bold')
                         
        axes[i].set_ylim(0, max(v_stat, v_dyn)*1.15)
        axes[i].tick_params(axis='x', labelsize=13)
        axes[i].tick_params(axis='y', labelsize=12)
        
    plt.tight_layout()
    plt.savefig('图2_宏观性能指标对比_SCI.png', bbox_inches='tight')
    
    # =========================================================
    # 图3: 微观调度特征分布 (SCI Raincloud/Violin Plots)
    # 根据用户偏好设计：使用Raincloud plot风格展示数据分布
    # =========================================================
    print("Generating Figure 3: Micro Feature Raincloud Plots...")
    fig, axes = plt.subplots(1, 2, figsize=(15, 6), dpi=300)
    fig.suptitle('微观调度特征分布特征 (Microscopic Feature Distribution)', fontsize=18, fontweight='bold', y=1.02)
    
    # 距离分布数据
    data_dist = pd.DataFrame({
        '距离': static_dist + dyn_dist,
        '方案': ['静态基线 (Static)']*len(static_dist) + ['动态重调度 (Dynamic)']*len(dyn_dist)
    })
    
    # 绘制距离的小提琴图+散点图 (Raincloud 效果)
    sns.violinplot(x='方案', y='距离', hue='方案', data=data_dist, ax=axes[0], 
                   palette=[NPG_COLORS[0], NPG_COLORS[1]], inner='box', legend=False, alpha=0.5)
    sns.stripplot(x='方案', y='距离', hue='方案', data=data_dist, ax=axes[0], 
                  palette=[NPG_COLORS[0], NPG_COLORS[1]], size=7, jitter=True, alpha=0.8, edgecolor='white', linewidth=1, legend=False)
    
    axes[0].set_ylabel('单车行驶距离 (km)', fontsize=14)
    axes[0].set_xlabel('')
    axes[0].set_title('(a) 单车行驶距离核密度与散点分布', fontsize=15, pad=15)
    axes[0].tick_params(axis='both', labelsize=13)
    
    # 载重分布数据
    data_load = pd.DataFrame({
        '载重': static_loads + dyn_loads,
        '方案': ['静态基线 (Static)']*len(static_loads) + ['动态重调度 (Dynamic)']*len(dyn_loads)
    })
    
    # 绘制载重的小提琴图+散点图 (Raincloud 效果)
    sns.violinplot(x='方案', y='载重', hue='方案', data=data_load, ax=axes[1], 
                   palette=[NPG_COLORS[0], NPG_COLORS[1]], inner='box', legend=False, alpha=0.5)
    sns.stripplot(x='方案', y='载重', hue='方案', data=data_load, ax=axes[1], 
                  palette=[NPG_COLORS[0], NPG_COLORS[1]], size=7, jitter=True, alpha=0.8, edgecolor='white', linewidth=1, legend=False)
    
    axes[1].set_ylabel('单车实际载重量 (kg)', fontsize=14)
    axes[1].set_xlabel('')
    axes[1].set_title('(b) 车辆装载容量核密度与散点分布', fontsize=15, pad=15)
    axes[1].tick_params(axis='both', labelsize=13)
    
    sns.despine()
    plt.tight_layout()
    plt.savefig('图3_微观分布特征_Raincloud_SCI.png', bbox_inches='tight')
    print("All visualizations generated successfully.")

if __name__ == "__main__":
    main()
