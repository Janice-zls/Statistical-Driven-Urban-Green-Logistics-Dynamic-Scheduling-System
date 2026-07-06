import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.patches as patches
from matplotlib.gridspec import GridSpec
import re

# =============================================================================
# 博士生级别（SCI/Nature风格）绘图参数设置
# =============================================================================
def setup_sci_nature_style():
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["SimHei", "Microsoft YaHei", "Arial"],
        "axes.unicode_minus": False,
        "axes.linewidth": 1.2,
        "axes.titlesize": 16,
        "axes.labelsize": 14,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "legend.fontsize": 12,
        "legend.frameon": False,
        "figure.figsize": (18, 12),
        "figure.dpi": 300
    })
    return ['#E64B35', '#4DBBD5', '#00A087', '#3C5488', '#F39B7F', '#8491B4', '#91D1C2', '#DC0000']

NPG_COLORS = setup_sci_nature_style()

# =============================================================================
# 解析函数复用
# =============================================================================
def time_str_to_float(time_str):
    if pd.isna(time_str): return np.nan
    h, m = map(int, time_str.split(':'))
    return h + m / 60.0

def parse_schedule_excel(filepath, coords_df):
    df = pd.read_excel(filepath)
    events = []
    for _, row in df.iterrows():
        v_id = row['车辆编号']
        v_type = row['车辆类型']
        is_ev = 1 if v_type in [4, 5] else 0
        path_str = row['到达时间节点']
        nodes = path_str.split(' -> ')
        for node_str in nodes:
            match = re.match(r'(\d+)\((.*?)\)', node_str)
            if match:
                node_id = int(match.group(1))
                time_val = time_str_to_float(match.group(2))
                
                if node_id == 0:
                    dist = 0; x, y = 0, 0
                else:
                    coord_row = coords_df[coords_df['ID'] == node_id]
                    if not coord_row.empty:
                        x = coord_row['X (km)'].values[0]
                        y = coord_row['Y (km)'].values[0]
                        dist = np.sqrt(x**2 + y**2)
                    else:
                        dist = 0; x, y = 0, 0
                
                events.append({
                    'v_id': v_id, 'v_type': v_type, 'is_ev': is_ev,
                    'node_id': node_id, 'time': time_val, 'dist': dist, 'x': x, 'y': y
                })
    return pd.DataFrame(events)

# =============================================================================
# 第一幅大图：3D时空柱状体与车辆甘特图 (Spatiotemporal & Gantt)
# =============================================================================
def generate_fig1_spatiotemporal(events_p2):
    fig = plt.figure(figsize=(20, 10))
    gs = GridSpec(1, 2, figure=fig, width_ratios=[1, 1.2], wspace=0.15)
    
    # -------------------------------------------------------------------------
    # 左图：3D时空禁行圆柱体 (3D Spatiotemporal Cylinder)
    # -------------------------------------------------------------------------
    ax1 = fig.add_subplot(gs[0, 0], projection='3d')
    
    # 画绿色禁行圆柱体 (r=10, z从8到16)
    z = np.linspace(8, 16, 50)
    theta = np.linspace(0, 2*np.pi, 50)
    theta_grid, z_grid = np.meshgrid(theta, z)
    x_grid = 10 * np.cos(theta_grid)
    y_grid = 10 * np.sin(theta_grid)
    ax1.plot_surface(x_grid, y_grid, z_grid, alpha=0.15, color=NPG_COLORS[0], edgecolor='none')
    
    # 画底层绿色配送区圆盘 (z=0)
    ax1.plot_surface(x_grid, y_grid, np.zeros_like(z_grid), alpha=0.05, color=NPG_COLORS[2], edgecolor='none')
    
    # 抽取3条EV和3条Fuel路线展示
    ev_vids = events_p2[(events_p2['is_ev']==1)]['v_id'].unique()[:3]
    fuel_vids = events_p2[(events_p2['is_ev']==0)]['v_id'].unique()[:5]
    
    for vid in ev_vids:
        path = events_p2[events_p2['v_id'] == vid].sort_values('time')
        ax1.plot(path['x'], path['y'], path['time'], color=NPG_COLORS[2], lw=2.5, marker='o', markersize=4, alpha=0.8)
    
    for vid in fuel_vids:
        path = events_p2[events_p2['v_id'] == vid].sort_values('time')
        ax1.plot(path['x'], path['y'], path['time'], color=NPG_COLORS[0], lw=2, linestyle='--', marker='^', markersize=4, alpha=0.8)
    
    # 图例代理
    from matplotlib.lines import Line2D
    custom_lines = [
        Line2D([0], [0], color=NPG_COLORS[0], lw=10, alpha=0.3, label='时空禁行柱体 (8:00-16:00, R<10)'),
        Line2D([0], [0], color=NPG_COLORS[2], lw=2.5, marker='o', label='新能源车 3D轨迹 (无视禁行)'),
        Line2D([0], [0], color=NPG_COLORS[0], lw=2, linestyle='--', marker='^', label='燃油车 3D轨迹 (规避/延迟)')
    ]
    ax1.legend(handles=custom_lines, loc='upper left', fontsize=11, frameon=True)
    
    ax1.set_xlabel('X 坐标 (km)', fontweight='bold')
    ax1.set_ylabel('Y 坐标 (km)', fontweight='bold')
    ax1.set_zlabel('时间 (时)', fontweight='bold')
    ax1.set_zlim(6, 24)
    ax1.set_xlim(-25, 25)
    ax1.set_ylim(-25, 25)
    ax1.view_init(elev=20, azim=45)
    ax1.set_title('(A) 三维时空管状禁行约束下的路径演化图', fontweight='bold', pad=20)
    
    # -------------------------------------------------------------------------
    # 右图：车辆调度高密度甘特图 (High-Density Gantt Chart)
    # -------------------------------------------------------------------------
    ax2 = fig.add_subplot(gs[0, 1])
    
    y_labels = []
    y_ticks = []
    current_y = 0
    
    # 排序车辆，让EV在上面，Fuel在下面
    sorted_vids = events_p2.groupby('v_id').first().sort_values('is_ev', ascending=False).index
    
    for vid in sorted_vids:
        path = events_p2[events_p2['v_id'] == vid]
        min_t = path['time'].min()
        max_t = path['time'].max()
        is_ev = path['is_ev'].iloc[0]
        color = NPG_COLORS[2] if is_ev else NPG_COLORS[3]
        
        ax2.hlines(current_y, min_t, max_t, color=color, lw=4, alpha=0.8)
        # 画节点
        ax2.scatter(path['time'], [current_y]*len(path), color='black', s=10, zorder=3)
        
        y_ticks.append(current_y)
        y_labels.append(f"V{vid}({'EV' if is_ev else 'Fuel'})")
        current_y -= 1
        
    # 灰色禁行背景区
    ax2.axvspan(8, 16, color='grey', alpha=0.15, label='限行时段 (仅限EV进入市中心)')
    
    # 设置Y轴（由于车辆太多，只显示部分Tick）
    step = max(1, len(y_ticks)//20)
    ax2.set_yticks(y_ticks[::step])
    ax2.set_yticklabels(y_labels[::step], fontsize=9)
    ax2.set_xlim(6, 24)
    ax2.set_xlabel('一天中的时间 (时)', fontweight='bold')
    ax2.set_title('(B) 混合车队调度生命周期甘特图 (Gantt Chart)', fontweight='bold', pad=20)
    
    custom_lines2 = [
        Line2D([0], [0], color=NPG_COLORS[2], lw=4, label='新能源车作业时段'),
        Line2D([0], [0], color=NPG_COLORS[3], lw=4, label='燃油车作业时段 (大量后置)'),
        patches.Patch(facecolor='grey', alpha=0.2, label='核心限行时段 (8:00-16:00)')
    ]
    ax2.legend(handles=custom_lines2, loc='upper left', frameon=True)
    ax2.grid(True, axis='x', linestyle=':', alpha=0.6)
    
    plt.suptitle("问题2 深入可视化：三维时空演化与调度生命周期", fontsize=22, fontweight='bold', y=0.98)
    plt.savefig('问题2/博士生级_时空演化与调度甘特图.png', bbox_inches='tight', dpi=300)
    plt.savefig('问题2/博士生级_时空演化与调度甘特图.pdf', bbox_inches='tight')
    plt.close()

# =============================================================================
# 第二幅大图：能耗动力学、车次流图与成本双环图 (Energy, Stream & Cost)
# =============================================================================
def generate_fig2_energy_cost(df_p1, df_p2, events_p2):
    fig = plt.figure(figsize=(20, 14))
    gs = GridSpec(2, 2, figure=fig, wspace=0.2, hspace=0.3)
    
    # -------------------------------------------------------------------------
    # (A) U型能耗曲线与工作点映射 (U-Curve & Operating Points)
    # -------------------------------------------------------------------------
    ax1 = fig.add_subplot(gs[0, 0])
    v_range = np.linspace(5, 70, 100)
    fpk = 0.0025 * v_range**2 - 0.2554 * v_range + 31.75
    epk = 0.0014 * v_range**2 - 0.12 * v_range + 36.19
    
    ax1.plot(v_range, fpk, color=NPG_COLORS[0], lw=3, label='燃油车能耗曲线 (FPK)')
    ax1.plot(v_range, epk, color=NPG_COLORS[2], lw=3, label='新能源车能耗曲线 (EPK)')
    
    # 标出三个速度点
    speeds = [9.8, 35.4, 55.3]
    labels = ['拥堵\n(9.8 km/h)', '一般\n(35.4 km/h)', '顺畅\n(55.3 km/h)']
    for s, l in zip(speeds, labels):
        ax1.axvline(s, color='grey', linestyle='--', alpha=0.5)
        ax1.text(s, 35, l, rotation=90, va='bottom', ha='right', fontweight='bold', alpha=0.7)
        # 散点
        f_val = 0.0025 * s**2 - 0.2554 * s + 31.75
        ax1.scatter(s, f_val, color=NPG_COLORS[0], s=100, zorder=5, edgecolor='black')
        e_val = 0.0014 * s**2 - 0.12 * s + 36.19
        ax1.scatter(s, e_val, color=NPG_COLORS[2], s=100, zorder=5, edgecolor='black')
    
    # 高亮燃油车在问题2中被迫大量落入的“一般”与“拥堵”区间
    ax1.axvspan(5, 40, color=NPG_COLORS[0], alpha=0.1, label='问题2燃油车被迫集中的低效区间')
    
    ax1.set_xlabel('行驶速度 (km/h)', fontweight='bold')
    ax1.set_ylabel('百公里能耗基准值', fontweight='bold')
    ax1.set_title('(A) 车辆U型能耗曲线与工况偏移 (Energy Dynamics)', fontweight='bold', pad=15)
    ax1.legend(loc='upper right')
    ax1.grid(True, linestyle=':', alpha=0.6)
    
    # -------------------------------------------------------------------------
    # (B) 混合车队并发作业流图 (Streamgraph / Stacked Area)
    # -------------------------------------------------------------------------
    ax2 = fig.add_subplot(gs[0, 1])
    bins = np.linspace(6, 24, 36)
    ev_counts = []
    fuel_counts = []
    
    for i in range(len(bins)-1):
        t_start, t_end = bins[i], bins[i+1]
        active_ev = events_p2[(events_p2['time']>=t_start) & (events_p2['time']<t_end) & (events_p2['is_ev']==1)]['v_id'].nunique()
        active_fuel = events_p2[(events_p2['time']>=t_start) & (events_p2['time']<t_end) & (events_p2['is_ev']==0)]['v_id'].nunique()
        ev_counts.append(active_ev)
        fuel_counts.append(active_fuel)
        
    x_centers = (bins[:-1] + bins[1:]) / 2
    ax2.stackplot(x_centers, ev_counts, fuel_counts, labels=['新能源车 (EV)', '燃油车 (Fuel)'], 
                  colors=[NPG_COLORS[2], NPG_COLORS[3]], alpha=0.8, edgecolor='white')
    
    # 叠加禁行区
    ax2.axvspan(8, 16, color='grey', alpha=0.2, label='市中心限行时段')
    
    ax2.set_xlabel('一天中的时间 (时)', fontweight='bold')
    ax2.set_ylabel('并发作业车辆数 (辆)', fontweight='bold')
    ax2.set_title('(B) 系统并发运力流图 (Active Fleet Streamgraph)', fontweight='bold', pad=15)
    ax2.legend(loc='upper left')
    ax2.set_xlim(6, 24)
    ax2.grid(True, axis='y', linestyle=':', alpha=0.6)
    
    # -------------------------------------------------------------------------
    # (C) 成本结构深度解剖 (Nested Donut Charts)
    # -------------------------------------------------------------------------
    def plot_donut(ax, df, title):
        fixed = df['固定成本(元)'].sum()
        travel = df['行驶与碳排放成本(元)'].sum()
        penalty = df['时间窗惩罚成本(元)'].sum()
        total = fixed + travel + penalty
        
        vals = [fixed, travel, penalty]
        labels = [f'固定启动\n({fixed/total*100:.1f}%)', f'行驶与碳排\n({travel/total*100:.1f}%)', f'时间窗惩罚\n({penalty/total*100:.1f}%)']
        colors = [NPG_COLORS[5], NPG_COLORS[6], NPG_COLORS[0]]
        explode = (0.05, 0.05, 0.1) if penalty > total*0.1 else (0.05, 0.05, 0.05)
        
        wedges, texts, autotexts = ax.pie(vals, labels=labels, colors=colors, autopct='%1.1f%%', 
                                          startangle=140, pctdistance=0.85, explode=explode,
                                          textprops=dict(color="black", fontweight='bold', fontsize=12))
        
        # 绘制中心白圆
        centre_circle = plt.Circle((0,0), 0.60, fc='white')
        ax.add_patch(centre_circle)
        
        ax.text(0, 0, f'总计\n¥{total:,.0f}', ha='center', va='center', fontsize=16, fontweight='bold', color=NPG_COLORS[3])
        ax.set_title(title, fontweight='bold', pad=15)
        
    ax3 = fig.add_subplot(gs[1, 0])
    plot_donut(ax3, df_p1, '(C) 无政策环境成本结构 (问题1)')
    
    ax4 = fig.add_subplot(gs[1, 1])
    plot_donut(ax4, df_p2, '(D) 限行政策环境成本结构 (问题2)')
    
    plt.suptitle("问题2 深入可视化：能耗动力学、运力流与经济成本解剖", fontsize=22, fontweight='bold', y=0.98)
    plt.savefig('问题2/博士生级_能耗动力学与成本结构图.png', bbox_inches='tight', dpi=300)
    plt.savefig('问题2/博士生级_能耗动力学与成本结构图.pdf', bbox_inches='tight')
    print("【真·Nature风】更多高级图表生成完毕！")

if __name__ == "__main__":
    coords = pd.read_excel('附件/客户坐标信息.xlsx')
    events_p2 = parse_schedule_excel('问题2/车辆调度方案.xlsx', coords)
    
    df_p1 = pd.read_excel('问题1/车辆调度方案.xlsx')
    df_p2 = pd.read_excel('问题2/车辆调度方案.xlsx')
    
    generate_fig1_spatiotemporal(events_p2)
    generate_fig2_energy_cost(df_p1, df_p2, events_p2)
