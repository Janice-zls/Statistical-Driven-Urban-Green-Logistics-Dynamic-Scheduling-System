import os
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.gridspec import GridSpec
import warnings
warnings.filterwarnings('ignore')

class HiddenPrints:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')
    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout

# 全局样式配置 (白底配置)
plt.style.use('default')
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['savefig.facecolor'] = '#FFFFFF'  # 白底
plt.rcParams['axes.facecolor'] = '#FFFFFF'
plt.rcParams['axes.edgecolor'] = '#333333'
plt.rcParams['text.color'] = '#000000'         # 黑字
plt.rcParams['axes.labelcolor'] = '#000000'
plt.rcParams['xtick.color'] = '#000000'
plt.rcParams['ytick.color'] = '#000000'
plt.rcParams['grid.color'] = '#cccccc'         # 浅灰网格

# 适配白底的高饱和度科技配色
CYBER_NEON = ['#0088aa', '#cc00aa', '#66aa00', '#aa3300', '#009944', '#6600aa']

# 导入底层模型
from phd_alns_dynamic_vrptw import DynamicVRPTW, AdvancedSolver, simulate_dynamic_events, get_speed, SERVICE_TIME

def extract_timelines(env, routes, vtypes):
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

def plot_stunning_digital_twin_1_white(env, timelines, output_path):
    fig = plt.figure(figsize=(24, 14), facecolor='#FFFFFF')
    fig.suptitle('【城市绿色物流数字孪生指挥舱】 - 时空联合演化分析 (Space-Time Evolution)', 
                 fontsize=28, fontweight='bold', color='#0055ff', y=0.98)
    
    gs = GridSpec(1, 2, figure=fig, width_ratios=[1, 1.2], wspace=0.15)
    
    # ----------------------------------------------------
    # 左图：3D 时空棱柱演化轨迹
    # ----------------------------------------------------
    ax1 = fig.add_subplot(gs[0, 0], projection='3d')
    ax1.set_facecolor('#FFFFFF')
    
    # 绘制基础网络平面 (Z=8:00) 和绿色环保区
    theta = np.linspace(0, 2*np.pi, 50)
    z = np.linspace(8, 24, 2)
    Theta, Z = np.meshgrid(theta, z)
    X_cyl = 10 * np.cos(Theta)
    Y_cyl = 10 * np.sin(Theta)
    ax1.plot_surface(X_cyl, Y_cyl, Z, color='#00cc55', alpha=0.05, edgecolor='none')
    
    # 绘制轨迹
    for idx, tl in enumerate(timelines):
        if idx > 15: continue
        color = CYBER_NEON[idx % len(CYBER_NEON)]
        xs = [env.coords[n][0] for n, arr, st, lv in tl['timeline']]
        ys = [env.coords[n][1] for n, arr, st, lv in tl['timeline']]
        zs = [lv for n, arr, st, lv in tl['timeline']]
        
        glow = [pe.Stroke(linewidth=5, foreground=color, alpha=0.3), pe.Normal()]
        ax1.plot(xs, ys, zs, color=color, linewidth=2.5, path_effects=glow, marker='o', markersize=5, zorder=3)
        
        for x, y, z_val in zip(xs, ys, zs):
            ax1.plot([x, x], [y, y], [8, z_val], color=color, linestyle=':', alpha=0.4)
            
    ax1.scatter([0], [0], [8], c='#cc00aa', s=200, marker='*', zorder=10, 
                path_effects=[pe.withStroke(linewidth=3, foreground='white')])
    
    ax1.set_title('(a) 3D时空棱柱演化轨迹 (Space-Time Prism)', fontsize=18, color='black', pad=20)
    ax1.set_xlabel('经度投影 (X)', fontsize=12, labelpad=10)
    ax1.set_ylabel('纬度投影 (Y)', fontsize=12, labelpad=10)
    ax1.set_zlabel('时间 (Time 8:00-24:00)', fontsize=12, labelpad=10)
    ax1.set_zlim(8, 24)
    ax1.view_init(elev=25, azim=-45)
    
    # ----------------------------------------------------
    # 右图：高清数字孪生甘特图
    # ----------------------------------------------------
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.set_facecolor('#FFFFFF')
    
    y_ticks = []
    y_labels = []
    
    for idx, tl in enumerate(timelines):
        if idx > 25: break
        y = len(timelines) - idx - 1 if len(timelines) <= 25 else 25 - idx - 1
        y_ticks.append(y)
        vtype = tl['vtype']
        y_labels.append(f"V{idx+1} [{vtype}]")
        
        color = CYBER_NEON[idx % len(CYBER_NEON)]
        route_timeline = tl['timeline']
        
        for i in range(1, len(route_timeline)):
            n_prev, _, _, lv_prev = route_timeline[i-1]
            n, arr, st, lv = route_timeline[i]
            
            # 行驶时间
            travel_dur = arr - lv_prev
            glow = [pe.Stroke(linewidth=3, foreground=color, alpha=0.4), pe.Normal()]
            ax2.barh(y, travel_dur, left=lv_prev, height=0.4, color=color, alpha=0.9, path_effects=glow)
            
            # 等待时间
            wait_dur = st - arr
            if wait_dur > 0:
                ax2.barh(y, wait_dur, left=arr, height=0.4, color='#999999', alpha=0.4, hatch='//')
                
            # 服务时间
            serv_dur = lv - st
            if serv_dur > 0:
                ax2.barh(y, serv_dur, left=st, height=0.6, color='white', alpha=0.9, edgecolor=color, linewidth=1.5)
                ax2.text(st + serv_dur/2, y, str(n), color='black', fontsize=7, fontweight='bold', 
                         ha='center', va='center', zorder=10)
    
    ax2.set_yticks(y_ticks)
    ax2.set_yticklabels(y_labels, fontsize=10, color='black')
    ax2.set_xlim(8, 24)
    ax2.set_xticks(np.arange(8, 25, 2))
    ax2.set_xticklabels([f"{int(h)}:00" for h in np.arange(8, 25, 2)], fontsize=12)
    ax2.set_title('(b) 高清数字孪生甘特图 (Digital Gantt Chart)', fontsize=18, color='black', pad=60)
    ax2.set_xlabel('当日时间 (Hour)', fontsize=14)
    ax2.grid(axis='x', color='#cccccc', linestyle='--', alpha=0.8)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.spines['left'].set_color('#333333')
    ax2.spines['bottom'].set_color('#333333')
    
    # 自定义图例
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=CYBER_NEON[0], label='行驶状态 (Driving)'),
        Patch(facecolor='#999999', alpha=0.4, hatch='//', label='等待时间窗 (Waiting)'),
        Patch(facecolor='white', edgecolor=CYBER_NEON[0], linewidth=1.5, label='客户交接服务 (Service)')
    ]
    # 将图例移动到图表正上方外部
    ax2.legend(handles=legend_elements, loc='lower center', bbox_to_anchor=(0.5, 1.02), ncol=3, facecolor='#FFFFFF', edgecolor='#cccccc', fontsize=12)
    
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()
    print(f"白底图已生成: {output_path}")

if __name__ == "__main__":
    print("Initializing Core Model Environment...")
    base_dir = r"g:\B.比赛\2026第十八届华中杯大学生数学建模挑战赛\A题城市绿色物流配送调度_1776844160973"
    env = DynamicVRPTW(os.path.join(base_dir, "附件"))
    solver = AdvancedSolver(env)
    
    print("Running Simulator silently...")
    with HiddenPrints():
        s_routes, s_vtypes = solver.build_initial_routes(range(1, env.num_nodes))
        d_routes, d_vtypes = simulate_dynamic_events(env, solver, s_routes, s_vtypes)
        
    print("Extracting detailed space-time timelines...")
    d_timelines = extract_timelines(env, d_routes, d_vtypes)
    
    # 覆盖用户要求修改底色的原图文件路径
    output_path = os.path.join(base_dir, "问题3", "惊艳_1_3D时空与甘特图_科技风.png")
    print(f"Generating White Background Image to: {output_path}")
    plot_stunning_digital_twin_1_white(env, d_timelines, output_path)
    print("All tasks completed successfully!")
