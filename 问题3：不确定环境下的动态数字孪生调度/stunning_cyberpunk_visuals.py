import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.gridspec import GridSpec
import warnings
warnings.filterwarnings('ignore')

# 抑制输出
class HiddenPrints:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')
    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout

# ---------------------------------------------------------
# 全局极致美学设置 (数字孪生/赛博朋克 科技大屏风)
# ---------------------------------------------------------
plt.style.use('dark_background')
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['savefig.facecolor'] = '#0a0d14'
plt.rcParams['axes.facecolor'] = '#0a0d14'
plt.rcParams['axes.edgecolor'] = '#334455'
plt.rcParams['text.color'] = '#e2e8f0'
plt.rcParams['axes.labelcolor'] = '#cbd5e1'
plt.rcParams['xtick.color'] = '#cbd5e1'
plt.rcParams['ytick.color'] = '#cbd5e1'
plt.rcParams['grid.color'] = '#1e293b'

# 赛博朋克调色板
CYBER_NEON = ['#00f3ff', '#ff00ea', '#ccff00', '#ff4500', '#00ff66', '#a700ff']

# 导入核心模块
from phd_alns_dynamic_vrptw import DynamicVRPTW, AdvancedSolver, simulate_dynamic_events, get_speed, SERVICE_TIME

def extract_timelines(env, routes, vtypes):
    """ 提取每一辆车的详细时空轨迹 (节点, 到达, 开始, 离开) """
    timelines = []
    for r, vt in zip(routes, vtypes):
        curr_time = 8.0
        route_timeline = []
        for i in range(len(r)):
            n = r[i]
            if i == 0:
                route_timeline.append((n, curr_time, curr_time, curr_time))
            else:
                prev_n = r[i-1]
                dist = env.dist_matrix[prev_n, n]
                speed = get_speed(curr_time)
                t_arrival = curr_time + dist / speed
                if n != 0:
                    tw_s, tw_e = env.time_windows[n]
                    t_start = max(t_arrival, tw_s)
                    t_leave = t_start + SERVICE_TIME
                else:
                    t_start = t_arrival
                    t_leave = t_arrival
                route_timeline.append((n, t_arrival, t_start, t_leave))
                curr_time = t_leave
        timelines.append({'vtype': vt, 'route': r, 'timeline': route_timeline})
    return timelines

def plot_stunning_digital_twin_1(env, timelines, output_path):
    """ 图1：3D时空轨迹立方体 (Space-Time Cube) + 智能调度甘特图 (Gantt) (1x2子图) """
    fig = plt.figure(figsize=(24, 11), facecolor='#0a0d14')
    fig.suptitle('【城市绿色物流数字孪生指挥舱】 - 时空联合演化分析 (Space-Time Evolution)', 
                 fontsize=28, fontweight='bold', color='#00f3ff', y=0.96)
    
    gs = GridSpec(1, 2, figure=fig, width_ratios=[1, 1.2], wspace=0.15)
    
    # -----------------------------
    # 子图 (a)：3D 时空轨迹 (X-Y-Time)
    # -----------------------------
    ax1 = fig.add_subplot(gs[0, 0], projection='3d')
    ax1.set_facecolor('#0a0d14')
    
    # 绘制绿色配送区 (3D 圆柱虚影)
    theta = np.linspace(0, 2*np.pi, 50)
    z = np.linspace(8, 24, 2)
    Theta, Z = np.meshgrid(theta, z)
    X_cyl = 10 * np.cos(Theta)
    Y_cyl = 10 * np.sin(Theta)
    ax1.plot_surface(X_cyl, Y_cyl, Z, color='#00ff66', alpha=0.05, edgecolor='none')
    
    # 绘制时空轨迹
    for idx, tl in enumerate(timelines):
        if idx > 15: continue # 只画前15辆避免过载
        color = CYBER_NEON[idx % len(CYBER_NEON)]
        xs = [env.coords[n][0] for n, arr, st, lv in tl['timeline']]
        ys = [env.coords[n][1] for n, arr, st, lv in tl['timeline']]
        zs = [lv for n, arr, st, lv in tl['timeline']]
        
        # 轨迹带发光特效
        glow = [pe.Stroke(linewidth=5, foreground=color, alpha=0.4), pe.Normal()]
        ax1.plot(xs, ys, zs, color=color, linewidth=2.5, path_effects=glow, marker='o', markersize=4, zorder=3)
        
        # 垂直投影线到地面 (Z=8)
        for x, y, z_val in zip(xs, ys, zs):
            ax1.plot([x, x], [y, y], [8, z_val], color=color, linestyle=':', alpha=0.3)
            
    # 中心点
    ax1.scatter([0], [0], [8], c='#ff00ea', s=200, marker='*', zorder=10, 
                path_effects=[pe.withStroke(linewidth=3, foreground='white')])
    
    ax1.set_title('(a) 3D时空棱柱演化轨迹 (Space-Time Prism)', fontsize=18, color='#cbd5e1', pad=20)
    ax1.set_xlabel('经度投影 (X)', fontsize=12, labelpad=10)
    ax1.set_ylabel('纬度投影 (Y)', fontsize=12, labelpad=10)
    ax1.set_zlabel('时间 (Time 8:00-24:00)', fontsize=12, labelpad=10)
    ax1.set_zlim(8, 24)
    ax1.view_init(elev=25, azim=-45) # 倾斜视角
    
    # -----------------------------
    # 子图 (b)：极客风甘特图 (Gantt Chart)
    # -----------------------------
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.set_facecolor('#0a0d14')
    
    y_ticks = []
    y_labels = []
    
    for idx, tl in enumerate(timelines):
        if idx > 25: break # 最多展示25辆车
        y = len(timelines) - idx - 1 if len(timelines) <= 25 else 25 - idx - 1
        y_ticks.append(y)
        vtype = tl['vtype']
        y_labels.append(f"V{idx+1} [{vtype}]")
        
        color = CYBER_NEON[idx % len(CYBER_NEON)]
        route_timeline = tl['timeline']
        
        for i in range(1, len(route_timeline)):
            n_prev, _, _, lv_prev = route_timeline[i-1]
            n, arr, st, lv = route_timeline[i]
            
            # 行驶时间 (实心)
            travel_dur = arr - lv_prev
            glow = [pe.Stroke(linewidth=3, foreground=color, alpha=0.5), pe.Normal()]
            ax2.barh(y, travel_dur, left=lv_prev, height=0.4, color=color, alpha=0.9, path_effects=glow)
            
            # 等待时间 (虚线框或浅色)
            wait_dur = st - arr
            if wait_dur > 0:
                ax2.barh(y, wait_dur, left=arr, height=0.4, color='#cbd5e1', alpha=0.3, hatch='//')
                
            # 服务时间 (亮色高光)
            serv_dur = lv - st
            if serv_dur > 0:
                ax2.barh(y, serv_dur, left=st, height=0.6, color='white', alpha=0.9, edgecolor=color)
                ax2.text(st + serv_dur/2, y, str(n), color='black', fontsize=7, fontweight='bold', 
                         ha='center', va='center', zorder=10)
    
    ax2.set_yticks(y_ticks)
    ax2.set_yticklabels(y_labels, fontsize=10, color='#cbd5e1')
    ax2.set_xlim(8, 24)
    ax2.set_xticks(np.arange(8, 25, 2))
    ax2.set_xticklabels([f"{int(h)}:00" for h in np.arange(8, 25, 2)], fontsize=12)
    ax2.set_title('(b) 高清数字孪生甘特图 (Digital Gantt Chart)', fontsize=18, color='#cbd5e1', pad=20)
    ax2.set_xlabel('当日时间 (Hour)', fontsize=14)
    ax2.grid(axis='x', color='#1e293b', linestyle='--', alpha=0.8)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.spines['left'].set_color('#334455')
    ax2.spines['bottom'].set_color('#334455')
    
    # 自定义图例
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=CYBER_NEON[0], label='行驶状态 (Driving)'),
        Patch(facecolor='#cbd5e1', alpha=0.3, hatch='//', label='等待时间窗 (Waiting)'),
        Patch(facecolor='white', edgecolor=CYBER_NEON[0], label='客户交接服务 (Service)')
    ]
    ax2.legend(handles=legend_elements, loc='upper right', facecolor='#0a0d14', edgecolor='#334455', fontsize=12)
    
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()

def plot_stunning_digital_twin_2(env, s_timelines, d_timelines, output_path):
    """ 图2：发光特效路网拓扑对比 (1x2子图，静态 vs 动态) """
    fig, axes = plt.subplots(1, 2, figsize=(22, 10), facecolor='#0a0d14', dpi=300)
    fig.suptitle('【城市绿色物流数字孪生指挥舱】 - 动态重连拓扑映射 (Topology Mapping)', 
                 fontsize=28, fontweight='bold', color='#00f3ff', y=1.02)
    
    titles = ['(a) 初始静态全局拓扑 (Static Network)', '(b) 事件驱动的动态重连拓扑 (Dynamic Re-wiring)']
    
    for ax, timelines, title, is_dynamic in zip(axes, [s_timelines, d_timelines], titles, [False, True]):
        ax.set_facecolor('#0a0d14')
        
        # 绿色区 (网格光晕)
        green_zone = plt.Circle((0, 0), 10, color='#00ff66', alpha=0.1, fill=True, zorder=1)
        ax.add_patch(green_zone)
        green_outline = plt.Circle((0, 0), 10, color='#00ff66', alpha=0.6, fill=False, linestyle='--', linewidth=2, zorder=2)
        ax.add_patch(green_outline)
        
        # 客户点 (星空散点)
        ax.scatter(env.coords[1:, 0], env.coords[1:, 1], c='#1e293b', s=30, edgecolors='#334455', zorder=3)
        
        # 绘制发光路径
        for idx, tl in enumerate(timelines):
            color = CYBER_NEON[idx % len(CYBER_NEON)]
            route = tl['route']
            xs = [env.coords[n][0] for n in route]
            ys = [env.coords[n][1] for n in route]
            
            # 是否受到突发事件影响而高亮 (客户15 和 99)
            is_affected = (15 in route or 99 in route) and is_dynamic
            
            if is_affected:
                glow = [pe.Stroke(linewidth=8, foreground=color, alpha=0.6), pe.Normal()]
                lw, alpha = 3.5, 1.0
            else:
                glow = [pe.Stroke(linewidth=4, foreground=color, alpha=0.2), pe.Normal()]
                lw, alpha = 1.5, 0.5
                
            ax.plot(xs, ys, color=color, linewidth=lw, alpha=alpha, path_effects=glow, zorder=4)
            
            if is_affected:
                ax.scatter(xs, ys, c='white', s=50, edgecolors=color, linewidth=2, zorder=5)
        
        # 配送中心
        ax.scatter([0], [0], c='#ff00ea', marker='p', s=600, zorder=10, 
                   path_effects=[pe.withStroke(linewidth=5, foreground='white')])
                   
        # 突发事件标注
        if is_dynamic:
            ax.scatter([env.coords[15][0]], [env.coords[15][1]], c='black', marker='X', s=400, edgecolors='red', linewidth=3, zorder=15, label='订单取消 (Node 15)')
            if len(env.coords) > 99:
                ax.scatter([env.coords[99][0]], [env.coords[99][1]], c='yellow', marker='*', s=600, edgecolors='red', linewidth=2, zorder=15, label='紧急插入 (Node 99)')
            ax.legend(facecolor='#0a0d14', edgecolor='#334455', fontsize=14, loc='upper right', textcolor='white')
            
        ax.set_title(title, fontsize=18, color='#cbd5e1', pad=20)
        ax.set_xlim([-35, 45])
        ax.set_ylim([-35, 45])
        ax.axis('off') # 去除所有轴线，彻底的科技大屏感
        
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    print("Initializing Core Model Environment...")
    env = DynamicVRPTW(r"../附件")
    solver = AdvancedSolver(env)
    
    print("Running Simulator silently...")
    with HiddenPrints():
        s_routes, s_vtypes = solver.build_initial_routes(range(1, env.num_nodes))
        d_routes, d_vtypes = simulate_dynamic_events(env, solver, s_routes, s_vtypes)
        
    print("Extracting detailed space-time timelines...")
    s_timelines = extract_timelines(env, s_routes, s_vtypes)
    d_timelines = extract_timelines(env, d_routes, d_vtypes)
    
    print("Generating [Image 1]: Cyberpunk Space-Time Prism & Gantt...")
    plot_stunning_digital_twin_1(env, d_timelines, '惊艳_1_3D时空与甘特图_科技风.png')
    
    print("Generating [Image 2]: Glowing Network Topology...")
    plot_stunning_digital_twin_2(env, s_timelines, d_timelines, '惊艳_2_发光路网拓扑_科技风.png')
    
    print("All Cyberpunk/Digital Twin visualizations completed successfully!")