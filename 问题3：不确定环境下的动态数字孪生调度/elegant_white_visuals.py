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

# ---------------------------------------------------------
# 全局极简美学设置 (干净/白底/高级学术风/商业风)
# ---------------------------------------------------------
# 绝对禁止黑底！
plt.style.use('default')
plt.rcParams['figure.facecolor'] = '#FFFFFF'
plt.rcParams['axes.facecolor'] = '#F8F9FA'
plt.rcParams['savefig.facecolor'] = '#FFFFFF'
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['axes.edgecolor'] = '#CBD5E1'
plt.rcParams['axes.linewidth'] = 1.2
plt.rcParams['grid.color'] = '#E2E8F0'
plt.rcParams['text.color'] = '#1E293B'
plt.rcParams['axes.labelcolor'] = '#334155'
plt.rcParams['xtick.color'] = '#475569'
plt.rcParams['ytick.color'] = '#475569'

# 高级现代商业调色板 (Tableau/Material 风格)
MODERN_COLORS = {
    'blue': '#2563EB',
    'green': '#10B981',
    'red': '#EF4444',
    'orange': '#F59E0B',
    'purple': '#8B5CF6',
    'gray': '#94A3B8'
}

# 抑制打印
class HiddenPrints:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')
    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout

from phd_alns_dynamic_vrptw import DynamicVRPTW, AdvancedSolver, simulate_dynamic_events, get_speed, SERVICE_TIME, VEHICLE_TYPES

def extract_node_arrivals(env, routes, vtypes):
    node_arrivals = {}
    node_tws = {}
    caps = []
    
    for v_idx, (r, vt) in enumerate(zip(routes, vtypes)):
        curr_time = 8.0
        w_sum = 0
        v_sum = 0
        for i in range(1, len(r)):
            prev_n = r[i-1]
            n = r[i]
            dist = env.dist_matrix[prev_n, n]
            speed = get_speed(curr_time)
            arr = curr_time + dist / speed
            
            if n != 0:
                tw_s, tw_e = env.time_windows[n]
                node_arrivals[n] = arr
                node_tws[n] = (tw_s, tw_e)
                st = max(arr, tw_s)
                curr_time = st + SERVICE_TIME
                w_sum += env.demands[n][0]
                v_sum += env.demands[n][1]
            else:
                curr_time = arr
                
        cap_w = VEHICLE_TYPES[vt]['cap_w']
        cap_v = VEHICLE_TYPES[vt]['cap_v']
        caps.append({
            'vid': f"V{v_idx+1}\n({vt})",
            'vtype': vt,
            'w_util': (w_sum / cap_w) * 100 if cap_w > 0 else 0,
            'v_util': (v_sum / cap_v) * 100 if cap_v > 0 else 0
        })
    return node_arrivals, node_tws, caps

def plot_elegant_dashboard_1(node_arrivals, node_tws, caps, output_path):
    """ 图1：客户时间窗依从度 (Scatter Range) & 车辆双容量满载率 (Grouped Bar) """
    fig, axes = plt.subplots(1, 2, figsize=(20, 8), dpi=300)
    fig.suptitle('物流系统精细化服务指标分析', fontsize=24, fontweight='bold', color=MODERN_COLORS['blue'], y=1.05)
    
    # -----------------------------
    # (a) 时间窗依从度分析 (Time Window Adherence)
    # -----------------------------
    ax1 = axes[0]
    
    # 按到达时间排序节点
    sorted_nodes = sorted(node_arrivals.keys(), key=lambda x: node_arrivals[x])
    y_pos = np.arange(len(sorted_nodes))
    
    # 绘制时间窗范围 (浅灰色粗线)
    for idx, n in enumerate(sorted_nodes):
        tw_s, tw_e = node_tws[n]
        arr = node_arrivals[n]
        
        # 背景时间窗
        ax1.hlines(idx, tw_s, tw_e, color='#E2E8F0', linewidth=4, zorder=1)
        
        # 惩罚区间标记 (红色虚线表示违规)
        color = MODERN_COLORS['blue']
        marker = 'o'
        if arr < tw_s:
            color = MODERN_COLORS['orange']
            marker = '<'
        elif arr > tw_e:
            color = MODERN_COLORS['red']
            marker = '>'
            
        # 实际到达时间点
        ax1.scatter(arr, idx, color=color, s=50, marker=marker, zorder=3, edgecolor='white', linewidth=0.5)
        
    ax1.set_title('(a) 客户节点时间窗满足度全景映射', fontsize=16, fontweight='bold', pad=15)
    ax1.set_xlabel('当日时刻 (Hour)', fontsize=14)
    ax1.set_ylabel('服务次序 (按到达时间排列)', fontsize=14)
    ax1.set_xlim([7.5, 24.5])
    ax1.set_xticks(np.arange(8, 25, 2))
    ax1.set_xticklabels([f"{int(h)}:00" for h in np.arange(8, 25, 2)], fontsize=12)
    ax1.set_yticks([]) # 隐藏 Y 轴具体刻度
    ax1.grid(axis='x', linestyle='--', alpha=0.5)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    
    # 自定义图例
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color='#E2E8F0', lw=4, label='客户规定时间窗'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor=MODERN_COLORS['blue'], markersize=8, label='准时到达'),
        Line2D([0], [0], marker='<', color='w', markerfacecolor=MODERN_COLORS['orange'], markersize=8, label='早到等待'),
        Line2D([0], [0], marker='>', color='w', markerfacecolor=MODERN_COLORS['red'], markersize=8, label='迟到惩罚')
    ]
    ax1.legend(handles=legend_elements, loc='lower right', fontsize=11, framealpha=0.9, edgecolor='#CBD5E1')
    
    # -----------------------------
    # (b) 车辆双维满载率 (Weight vs Volume)
    # -----------------------------
    ax2 = axes[1]
    
    x = np.arange(len(caps))
    w_utils = [c['w_util'] for c in caps]
    v_utils = [c['v_util'] for c in caps]
    v_labels = [c['vid'] for c in caps]
    
    width = 0.35
    bars1 = ax2.bar(x - width/2, w_utils, width, label='重量满载率 (Weight Util %)', color=MODERN_COLORS['blue'], alpha=0.85)
    bars2 = ax2.bar(x + width/2, v_utils, width, label='体积满载率 (Volume Util %)', color=MODERN_COLORS['green'], alpha=0.85)
    
    # 100% 警戒线
    ax2.axhline(100, color=MODERN_COLORS['red'], linestyle='--', linewidth=1.5, zorder=0, label='100% 容量红线')
    
    ax2.set_title('(b) 各调度车辆双维容量利用率分析', fontsize=16, fontweight='bold', pad=15)
    ax2.set_ylabel('满载率 (%)', fontsize=14)
    ax2.set_xticks(x)
    ax2.set_xticklabels(v_labels, fontsize=10, rotation=45)
    ax2.set_ylim([0, max(max(w_utils), max(v_utils)) * 1.15])
    ax2.grid(axis='y', linestyle='--', alpha=0.5)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=3, fontsize=12, frameon=False)
    
    # 添加数值标签
    for rects in [bars1, bars2]:
        for rect in rects:
            height = rect.get_height()
            if height > 0:
                ax2.annotate(f'{height:.0f}%',
                            xy=(rect.get_x() + rect.get_width() / 2, height),
                            xytext=(0, 3),  # 3 points vertical offset
                            textcoords="offset points",
                            ha='center', va='bottom', fontsize=9, color='#475569')
                            
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()

def plot_elegant_dashboard_2(df_details, total_cost, output_path):
    """ 图2：多维成本瀑布图 (Waterfall) & 异构车型综合性能雷达图 (Radar) """
    fig = plt.figure(figsize=(20, 8), dpi=300)
    fig.suptitle('调度成本溯源与车型效能评估', fontsize=24, fontweight='bold', color=MODERN_COLORS['blue'], y=1.05)
    
    gs = GridSpec(1, 2, width_ratios=[1.2, 1], wspace=0.3)
    
    # -----------------------------
    # (a) 成本瀑布图 (Cost Waterfall Chart)
    # -----------------------------
    ax1 = fig.add_subplot(gs[0, 0])
    
    df = pd.DataFrame(df_details)
    c_start = 500 # 预估或传入
    c_energy = df['energy_cost'].sum()
    c_tw = df['tw_cost'].sum() - c_start
    c_carbon = df['carbon_cost'].sum()
    c_total = c_start + c_energy + c_tw + c_carbon
    
    values = [c_start, c_energy, c_tw, c_carbon, c_total]
    categories = ['车辆启动(Base)', '+ 能耗成本(Energy)', '+ 时间窗惩罚(TW)', '+ 碳排成本(Carbon)', '总调度成本(Total)']
    
    # 计算阶梯
    bottoms = [0, c_start, c_start+c_energy, c_start+c_energy+c_tw, 0]
    colors = [MODERN_COLORS['gray'], MODERN_COLORS['blue'], MODERN_COLORS['orange'], MODERN_COLORS['green'], MODERN_COLORS['red']]
    
    x = np.arange(len(categories))
    bars = ax1.bar(x, values, bottom=bottoms, color=colors, width=0.6, alpha=0.9, edgecolor='white', linewidth=1.5)
    
    # 阶梯连线
    for i in range(1, len(categories)-1):
        ax1.plot([x[i-1]-0.3, x[i]+0.3], [bottoms[i], bottoms[i]], color='#94A3B8', linestyle='--', linewidth=1)
    
    ax1.set_title('(a) 调度系统多维成本瀑布溯源 (Cost Breakdown Waterfall)', fontsize=16, fontweight='bold', pad=15)
    ax1.set_ylabel('成本金额 (元)', fontsize=14)
    ax1.set_xticks(x)
    ax1.set_xticklabels(categories, fontsize=12, rotation=15)
    ax1.grid(axis='y', linestyle='--', alpha=0.5)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    
    # 添加数值标签
    for i, rect in enumerate(bars):
        height = rect.get_height()
        bottom = rect.get_y()
        val = values[i]
        ax1.text(rect.get_x() + rect.get_width()/2, bottom + height + (c_total*0.02), 
                 f"{val:.1f}", ha='center', va='bottom', fontweight='bold', color=colors[i], fontsize=12)

    # -----------------------------
    # (b) 异构车型性能雷达图 (Radar Chart)
    # -----------------------------
    ax2 = fig.add_subplot(gs[0, 1], polar=True)
    
    # 数据聚合
    v_types = ['E1', 'E2', 'F1', 'F2', 'F3']
    metrics = ['平均距离(km)', '平均能耗(元)', '碳排水平(元)', '平均时间惩罚(元)', '单车总成本(元)']
    
    df['actual_tw'] = df['tw_cost'] - 500
    df['v_cat'] = df['vtype']
    
    # 分组求平均
    radar_data = {}
    for vt in v_types:
        sub = df[df['vtype'] == vt]
        if len(sub) > 0:
            radar_data[vt] = [
                sub['dist'].mean(),
                sub['energy_cost'].mean(),
                sub['carbon_cost'].mean(),
                sub['actual_tw'].mean(),
                sub['total'].mean()
            ]
        else:
            radar_data[vt] = [0, 0, 0, 0, 0]
            
    # 归一化处理 (除以最大值，使范围在 0-1)
    max_vals = [max([radar_data[vt][i] for vt in v_types]) for i in range(len(metrics))]
    max_vals = [v if v > 0 else 1 for v in max_vals]
    
    angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
    angles += angles[:1] # 闭合
    
    # 绘制纯净风格雷达
    colors_radar = [MODERN_COLORS['green'], MODERN_COLORS['blue'], MODERN_COLORS['red'], MODERN_COLORS['orange'], MODERN_COLORS['purple']]
    for idx, vt in enumerate(v_types):
        if sum(radar_data[vt]) == 0: continue
        norm_vals = [radar_data[vt][i] / max_vals[i] for i in range(len(metrics))]
        norm_vals += norm_vals[:1]
        
        ax2.plot(angles, norm_vals, color=colors_radar[idx], linewidth=2, label=vt)
        ax2.fill(angles, norm_vals, color=colors_radar[idx], alpha=0.15)
        
    ax2.set_theta_offset(np.pi / 2)
    ax2.set_theta_direction(-1)
    ax2.set_thetagrids(np.degrees(angles[:-1]), metrics, fontsize=12)
    ax2.set_rlabel_position(0)
    ax2.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax2.set_yticklabels(["", "", "", ""], color="grey", size=8) # 隐藏极径刻度数字，保持干净
    ax2.set_ylim(0, 1.1)
    ax2.spines['polar'].set_visible(False)
    ax2.grid(color='#E2E8F0', linestyle='--', linewidth=1.2)
    
    ax2.set_title('(b) 各车型综合性能对比 (归一化雷达图)', fontsize=16, fontweight='bold', pad=30)
    ax2.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=11, framealpha=0.9, edgecolor='#CBD5E1', title='使用车型')
    
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    print("Running Elegant White Theme Visualizations...")
    
    # 初始化环境
    env = DynamicVRPTW(r"../附件")
    solver = AdvancedSolver(env)
    
    with HiddenPrints():
        # 获取动态调度结果
        s_routes, s_vtypes = solver.build_initial_routes(range(1, env.num_nodes))
        d_routes, d_vtypes = simulate_dynamic_events(env, solver, s_routes, s_vtypes)
        d_cost, d_details = solver.evaluate_routes(d_routes, d_vtypes)
    
    # 提取时间窗和容量数据
    node_arrivals, node_tws, caps = extract_node_arrivals(env, d_routes, d_vtypes)
    
    print("Generating Image 1: Time Window & Capacity...")
    plot_elegant_dashboard_1(node_arrivals, node_tws, caps, '绝美极简_1_服务指标分析.png')
    
    print("Generating Image 2: Cost Waterfall & Performance Radar...")
    plot_elegant_dashboard_2(d_details, d_cost, '绝美极简_2_成本瀑布与雷达.png')
    
    print("All bright, clean, elegant visualizations are ready!")