import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.gridspec import GridSpec
import matplotlib.patches as patches

# =============================================================================
# 博士生级别（SCI/Nature风格）绘图参数设置
# =============================================================================
def setup_sci_nature_style():
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["SimHei", "Microsoft YaHei", "Arial"],
        "axes.unicode_minus": False,
        "axes.linewidth": 1.5,
        "axes.titlesize": 16,
        "axes.labelsize": 14,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "legend.fontsize": 12,
        "legend.frameon": False,
        "figure.figsize": (20, 10),
        "figure.dpi": 300
    })
    return ['#E64B35', '#4DBBD5', '#00A087', '#3C5488', '#F39B7F', '#8491B4', '#91D1C2', '#DC0000']

NPG_COLORS = setup_sci_nature_style()

# 车辆参数表 (Type: Capacity_Weight, Capacity_Volume)
VEHICLE_CAPACITY = {
    1: (3000, 13.5),
    2: (1500, 10.8),
    3: (1250, 6.5),
    4: (3000, 15.0),
    5: (1250, 8.5)
}

import re

def time_str_to_float(t_str):
    if not isinstance(t_str, str):
        return float(t_str)
    if ':' in t_str:
        h, m = map(int, t_str.split(':'))
        return h + m / 60.0
    return float(t_str)

def parse_solution(json_path, excel_path, coords_df, tw_df):
    # 解析 Excel 以获取真实的时间和成本轨迹
    df = pd.read_excel(excel_path)
    events_cost = []
    events_penalty = []
    node_penalties = {}
    
    for _, row in df.iterrows():
        v_id = row['车辆编号']
        path_str = row['到达时间节点']
        nodes = path_str.split(' -> ')
        
        # 记录固定启动成本
        start_time_str = re.match(r'\d+\((.*?)\)', nodes[0]).group(1) if re.match(r'\d+\((.*?)\)', nodes[0]) else '08:00'
        start_t = time_str_to_float(start_time_str)
        events_cost.append((start_t, 400))
        
        for node_str in nodes:
            match = re.match(r'(\d+)\((.*?)\)', node_str)
            if match:
                node_id = int(match.group(1))
                time_val = time_str_to_float(match.group(2))
                
                if node_id != 0:
                    tw_row = tw_df[tw_df['客户编号'] == node_id]
                    if not tw_row.empty:
                        tw_end_str = tw_row['结束时间'].values[0]
                        tw_end = time_str_to_float(tw_end_str)
                        delay = max(0, time_val - tw_end)
                        pen = delay * 50
                        if pen > 0:
                            events_penalty.append((time_val, pen))
                            node_penalties[node_id] = node_penalties.get(node_id, 0) + pen
                            
        # 终点记录行驶与碳排放成本 (简化加在终点)
        end_time_str = re.match(r'\d+\((.*?)\)', nodes[-1]).group(1) if re.match(r'\d+\((.*?)\)', nodes[-1]) else '24:00'
        end_t = time_str_to_float(end_time_str)
        events_cost.append((end_t, row['行驶与碳排放成本(元)']))
        events_cost.append((end_t, row['时间窗惩罚成本(元)'])) # 宏观累加

    # 解析 JSON 获取装载率
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        
    utilization = []
    for r in data['routes']:
        v_type = r['vehicle']
        is_ev = 1 if v_type in [4, 5] else 0
        cap_w, cap_v = VEHICLE_CAPACITY[v_type]
        
        route_w = sum([d.get('w', 0) for d in r.get('demands', [])])
        route_v = sum([d.get('v', 0) for d in r.get('demands', [])])
        util_w = route_w / cap_w * 100
        util_v = route_v / cap_v * 100
        utilization.append({'v_type': v_type, 'is_ev': is_ev, 'util_w': util_w, 'util_v': util_v})
        
    df_cost = pd.DataFrame(events_cost, columns=['time', 'cost']).sort_values('time')
    df_cost['cum_cost'] = df_cost['cost'].cumsum()
    
    df_pen = pd.DataFrame(events_penalty, columns=['time', 'penalty']).sort_values('time')
    df_pen['cum_pen'] = df_pen['penalty'].cumsum() if not df_pen.empty else pd.Series([], dtype=float)
    
    geo_pen = []
    for _, row in coords_df.iterrows():
        nid = row['ID']
        if nid == 0: continue
        x, y = row['X (km)'], row['Y (km)']
        dist = np.sqrt(x**2 + y**2)
        geo_pen.append({
            'id': nid, 'x': x, 'y': y, 'dist': dist, 
            'penalty': node_penalties.get(nid, 0),
            'is_green': 1 if dist <= 10 else 0
        })
        
    return df_cost, df_pen, pd.DataFrame(utilization), pd.DataFrame(geo_pen)

def generate_ultimate_masterpiece():
    coords = pd.read_excel('附件/客户坐标信息.xlsx')
    tw_df = pd.read_excel('附件/时间窗.xlsx')
    
    c1, p1, u1, g1 = parse_solution('问题1/solution_opt.json', '问题1/车辆调度方案.xlsx', coords, tw_df)
    c2, p2, u2, g2 = parse_solution('问题2/solution_opt.json', '问题2/车辆调度方案.xlsx', coords, tw_df)
    
    fig = plt.figure(figsize=(22, 10))
    gs = GridSpec(1, 3, figure=fig, width_ratios=[1.2, 1, 1], wspace=0.25)
    
    # -------------------------------------------------------------------------
    # (A) 成本爆发拐点：累积轨迹曲线 (Cumulative Trajectory)
    # -------------------------------------------------------------------------
    ax1 = fig.add_subplot(gs[0, 0])
    
    ax1.plot(c1['time'], c1['cum_cost']/10000, color=NPG_COLORS[3], lw=4, label='问题1 总成本轨迹')
    ax1.plot(c2['time'], c2['cum_cost']/10000, color=NPG_COLORS[0], lw=4, label='问题2 总成本轨迹 (限行引爆)')
    
    if not p1.empty:
        ax1.plot(p1['time'], p1['cum_pen']/10000, color=NPG_COLORS[3], lw=2, linestyle='--', label='问题1 惩罚成本')
    if not p2.empty:
        ax1.plot(p2['time'], p2['cum_pen']/10000, color=NPG_COLORS[0], lw=2, linestyle='--', label='问题2 惩罚成本')
        
    # 标出禁行时段
    ax1.axvspan(8, 16, color='grey', alpha=0.15, label='限行时段 (能量积蓄)')
    
    # 标出“大爆发”拐点
    ax1.axvline(16, color='red', linestyle=':', lw=2)
    ax1.text(16.2, 3, '16:00 解禁\n燃油车集中爆发\n引发成本直线上升', color='red', fontweight='bold')
    
    ax1.set_xlabel('一天中的时间 (时)', fontweight='bold')
    ax1.set_ylabel('累积成本 (万元)', fontweight='bold')
    ax1.set_title('(A) 系统成本演化的“拐点爆发”效应 (Tipping Point)', fontweight='bold', pad=15)
    ax1.legend(loc='upper left')
    ax1.grid(True, linestyle=':', alpha=0.6)
    ax1.set_xlim(6, 24)
    
    # -------------------------------------------------------------------------
    # (B) 地理惩罚气泡图：痛点空间分布 (Spatial Pain Point Bubble Map)
    # -------------------------------------------------------------------------
    ax2 = fig.add_subplot(gs[0, 1])
    
    # 绘制绿色配送区
    green_zone = patches.Circle((0, 0), 10, color=NPG_COLORS[2], alpha=0.1, label='绿色配送区')
    ax2.add_patch(green_zone)
    ax2.plot(0, 0, marker='*', markersize=15, color='gold', markeredgecolor='black', zorder=5)
    
    # 绘制问题2的惩罚气泡
    # 将惩罚金额映射为气泡大小 (避免0大小)
    sizes = (g2['penalty'] / 50 + 1) * 15 
    colors = [NPG_COLORS[0] if p > 0 else 'grey' for p in g2['penalty']]
    alphas = [0.8 if p > 0 else 0.3 for p in g2['penalty']]
    
    scatter = ax2.scatter(g2['x'], g2['y'], s=sizes, c=colors, alpha=alphas, edgecolors='white', linewidth=0.5)
    
    # 手动图例
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor=NPG_COLORS[0], markersize=15, label='产生严重延误的客户点'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='grey', markersize=8, label='按时送达的客户点'),
        patches.Patch(facecolor=NPG_COLORS[2], alpha=0.1, label='绿色配送区')
    ]
    ax2.legend(handles=legend_elements, loc='upper left')
    
    ax2.set_aspect('equal')
    ax2.set_xlim(-25, 25)
    ax2.set_ylim(-25, 25)
    ax2.set_xlabel('X 坐标 (km)', fontweight='bold')
    ax2.set_ylabel('Y 坐标 (km)', fontweight='bold')
    ax2.set_title('(B) 政策次生伤害地理映射 (Delay Hotspots)', fontweight='bold', pad=15)
    ax2.grid(True, linestyle=':', alpha=0.6)
    
    # -------------------------------------------------------------------------
    # (C) 运力装载效率核密度图 (Capacity Utilization Ridge/Density)
    # -------------------------------------------------------------------------
    ax3 = fig.add_subplot(gs[0, 2])
    
    u2_ev = u2[u2['is_ev']==1]['util_v']
    u2_fuel = u2[u2['is_ev']==0]['util_v']
    
    sns.kdeplot(u2_ev, ax=ax3, color=NPG_COLORS[2], fill=True, alpha=0.5, linewidth=2, label='新能源车 (极限压榨)')
    sns.kdeplot(u2_fuel, ax=ax3, color=NPG_COLORS[0], fill=True, alpha=0.5, linewidth=2, label='燃油车 (低效补偿)')
    
    # 画均值线
    ax3.axvline(u2_ev.mean(), color=NPG_COLORS[2], linestyle='--', lw=2)
    ax3.axvline(u2_fuel.mean(), color=NPG_COLORS[0], linestyle='--', lw=2)
    
    ax3.set_xlabel('车辆体积装载率 (%)', fontweight='bold')
    ax3.set_ylabel('密度频率 (Density)', fontweight='bold')
    ax3.set_title('(C) 混合车队运力极化分布 (Capacity Utilization)', fontweight='bold', pad=15)
    ax3.legend(loc='upper left')
    ax3.grid(True, linestyle=':', alpha=0.6)
    ax3.set_xlim(0, 110)
    
    # 移除顶右边框
    for ax in [ax1, ax2, ax3]:
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
    plt.suptitle("终极压轴图：成本爆发拐点、地理延误热点与运力极化剖析", fontsize=24, fontweight='bold', y=1.02)
    plt.savefig('问题2/博士生级_终极压轴图_系统效率与痛点全景.png', bbox_inches='tight', dpi=300)
    plt.savefig('问题2/博士生级_终极压轴图_系统效率与痛点全景.pdf', bbox_inches='tight')
    print("【终极·Nature风】压轴图表生成完毕！")

if __name__ == "__main__":
    generate_ultimate_masterpiece()
