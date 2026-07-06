import pandas as pd
import numpy as np
import os
import random
import copy
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import matplotlib
import sys

# Set visualization style
matplotlib.use('Agg')
plt.style.use('seaborn-v0_8-paper')
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial']
matplotlib.rcParams['axes.unicode_minus'] = False
matplotlib.rcParams['figure.dpi'] = 300

# Params
base_path = r"g:\B.比赛\2026第十八届“华中杯”大学生数学建模挑战赛\A题：城市绿色物流配送调度_1776844160973\附件"
out_path = r"g:\B.比赛\2026第十八届“华中杯”大学生数学建模挑战赛\A题：城市绿色物流配送调度_1776844160973\问题1"
os.makedirs(out_path, exist_ok=True)

# 1. Load Data
df_orders = pd.read_excel(os.path.join(base_path, "订单信息.xlsx"))
df_dist = pd.read_excel(os.path.join(base_path, "距离矩阵.xlsx"))
df_coords = pd.read_excel(os.path.join(base_path, "客户坐标信息.xlsx"))
df_tw = pd.read_excel(os.path.join(base_path, "时间窗.xlsx"))

# Aggregate orders
customer_demands = df_orders.groupby("目标客户编号")[['重量', '体积']].sum().reset_index()

# Distance matrix (numpy array)
dist_matrix = df_dist.iloc[:, 1:].values # shape (99, 99), index 0 is depot

# Time windows
def time_to_hours(t_str):
    if pd.isna(t_str): return 0
    t_str = str(t_str)
    if ':' in t_str:
        h, m = t_str.split(':')
        return int(h) + int(m)/60.0
    return 0

tw_start = np.zeros(99)
tw_end = np.zeros(99)
for idx, row in df_tw.iterrows():
    c_id = int(row['客户编号'])
    tw_start[c_id] = time_to_hours(row['开始时间'])
    tw_end[c_id] = time_to_hours(row['结束时间'])
tw_end[tw_end == 0] = 24.0 # default if missing

# Demands
weight = np.zeros(99)
volume = np.zeros(99)
for idx, row in customer_demands.iterrows():
    c_id = int(row['目标客户编号'])
    weight[c_id] = row['重量']
    volume[c_id] = row['体积']

# Vehicle info
# type_id: (name, weight_cap, vol_cap, count, is_ev)
vehicles_info = [
    ('E1', 3000, 15.0, 10, True),
    ('E2', 1250, 8.5, 15, True),
    ('F1', 3000, 13.5, 60, False),
    ('F2', 1500, 10.8, 50, False),
    ('F3', 1250, 6.5, 50, False)
]

# Costs
START_COST = 400
WAIT_COST = 20
LATE_COST = 50
SERVICE_TIME = 20 / 60.0

def get_speed(t):
    t_mod = t % 24
    if 8 <= t_mod < 9 or 11.5 <= t_mod < 13:
        return 9.8
    elif 9 <= t_mod < 10 or 13 <= t_mod < 15:
        return 55.3
    elif 10 <= t_mod < 11.5 or 15 <= t_mod < 17:
        return 35.4
    else:
        return 35.4 # Default

def calc_energy_cost(v, dist, is_ev, load_ratio):
    # load_ratio = current_load / capacity
    if is_ev:
        epk = 0.0014 * v**2 - 0.12 * v + 36.19
        cost_per_km = (epk / 100) * (1.64 + 0.65 * 0.961)
        # full load penalty
        penalty = 1.0 + 0.35 * load_ratio
        return cost_per_km * dist * penalty
    else:
        fpk = 0.0025 * v**2 - 0.2554 * v + 31.75
        cost_per_km = (fpk / 100) * (7.61 + 0.65 * 2.547)
        penalty = 1.0 + 0.40 * load_ratio
        return cost_per_km * dist * penalty

def eval_route(route, v_type, v_w_cap, v_v_cap, is_ev, start_time=8.0):
    # route is list of customers [c1, c2, ...] without depot
    if not route:
        return 0, 0, True, []
    
    # check capacity
    total_w = sum(weight[c] for c in route)
    total_v = sum(volume[c] for c in route)
    if total_w > v_w_cap or total_v > v_v_cap:
        return float('inf'), 0, False, []
    
    t = start_time
    cost = START_COST
    current_w = total_w # start with full load
    
    # Depot to first
    curr = 0
    timeline = [(0, t)]
    for nxt in route:
        dist = dist_matrix[curr, nxt]
        v = get_speed(t)
        travel_time = dist / v
        # add energy cost
        load_ratio = current_w / v_w_cap
        cost += calc_energy_cost(v, dist, is_ev, load_ratio)
        
        t += travel_time
        # Time window
        if t < tw_start[nxt]:
            cost += WAIT_COST * (tw_start[nxt] - t)
            t = tw_start[nxt]
        elif t > tw_end[nxt]:
            cost += LATE_COST * (t - tw_end[nxt])
        
        timeline.append((nxt, t))
        t += SERVICE_TIME
        current_w -= weight[nxt]
        curr = nxt
        
    # Last to depot
    dist = dist_matrix[curr, 0]
    v = get_speed(t)
    travel_time = dist / v
    cost += calc_energy_cost(v, dist, is_ev, 0)
    t += travel_time
    timeline.append((0, t))
    
    return cost, t, True, timeline

# Construction heuristic (Savings/Greedy)
def solve():
    unassigned = set(range(1, 99))
    routes = []
    
    # Available vehicles
    available = []
    for vt in vehicles_info:
        for _ in range(vt[3]):
            available.append(vt)
            
    while unassigned and available:
        best_route = None
        best_cost = float('inf')
        best_vt = None
        best_timeline = None
        best_vt_idx = -1
        
        # Try to build a route greedily
        for idx, vt in enumerate(available):
            # To speed up, just test the first vehicle of each type
            if idx > 0 and available[idx] == available[idx-1]:
                continue
                
            curr_route = []
            curr_unassigned = list(unassigned)
            curr_w = 0
            curr_v = 0
            
            while curr_unassigned:
                best_next = -1
                best_add_cost = float('inf')
                for c in curr_unassigned:
                    if curr_w + weight[c] <= vt[1] and curr_v + volume[c] <= vt[2]:
                        test_route = curr_route + [c]
                        cost, _, valid, tl = eval_route(test_route, vt[0], vt[1], vt[2], vt[4])
                        if valid and cost < best_add_cost:
                            best_add_cost = cost
                            best_next = c
                
                if best_next != -1:
                    curr_route.append(best_next)
                    curr_w += weight[best_next]
                    curr_v += volume[best_next]
                    curr_unassigned.remove(best_next)
                else:
                    break
                    
            if curr_route:
                cost, _, valid, tl = eval_route(curr_route, vt[0], vt[1], vt[2], vt[4])
                if cost < best_cost:
                    best_cost = cost
                    best_route = curr_route
                    best_vt = vt
                    best_timeline = tl
                    best_vt_idx = idx
                    
        if best_route:
            routes.append({
                'v_type': best_vt[0],
                'route': best_route,
                'cost': best_cost,
                'timeline': best_timeline
            })
            for c in best_route:
                unassigned.remove(c)
            available.pop(best_vt_idx)
        else:
            print("Cannot assign remaining customers!")
            break
            
    return routes

print("Solving...")
routes = solve()
total_cost = sum(r['cost'] for r in routes)
print(f"Total Cost: {total_cost:.2f}")

# Save results
res = []
for i, r in enumerate(routes):
    path_str = "0-" + "-".join(map(str, r['route'])) + "-0"
    tl_str = ", ".join([f"{c}({t:.2f}h)" for c, t in r['timeline']])
    res.append({
        'Vehicle_ID': i+1,
        'Type': r['v_type'],
        'Route': path_str,
        'Cost': r['cost'],
        'Timeline': tl_str
    })
df_res = pd.DataFrame(res)
df_res.to_excel(os.path.join(out_path, "问题1_车辆调度方案.xlsx"), index=False)
print("Saved to 问题1_车辆调度方案.xlsx")

# Visualization
plt.figure(figsize=(10, 10))
plt.scatter(df_coords['X (km)'], df_coords['Y (km)'], c='blue', label='Customer', s=20)
plt.scatter(df_coords.iloc[0]['X (km)'], df_coords.iloc[0]['Y (km)'], c='red', marker='*', s=200, label='Depot')
for r in routes:
    route = [0] + r['route'] + [0]
    x = [df_coords.iloc[c]['X (km)'] for c in route]
    y = [df_coords.iloc[c]['Y (km)'] for c in route]
    plt.plot(x, y, alpha=0.5)
plt.legend()
plt.title(f"问题1 车辆调度路径 (总成本: {total_cost:.2f})")
plt.savefig(os.path.join(out_path, "问题1_路径图.png"))
print("Saved plot.")
