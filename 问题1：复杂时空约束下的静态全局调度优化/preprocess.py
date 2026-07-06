import pandas as pd
import numpy as np

def process_data():
    base_dir = "g:\\B.比赛\\2026第十八届华中杯大学生数学建模挑战赛\\A题城市绿色物流配送调度_1776844160973\\附件\\"
    
    # 1. 订单信息
    orders = pd.read_excel(base_dir + "订单信息.xlsx")
    demands = orders.groupby("目标客户编号")[['重量', '体积']].sum().reset_index()
    
    # 2. 坐标信息
    coords = pd.read_excel(base_dir + "客户坐标信息.xlsx")
    
    # 3. 时间窗
    tw = pd.read_excel(base_dir + "时间窗.xlsx")
    def time_to_hours(t_str):
        h, m = map(int, str(t_str).split(':'))
        return h + m / 60.0
    
    tw['start_h'] = tw['开始时间'].apply(time_to_hours)
    tw['end_h'] = tw['结束时间'].apply(time_to_hours)
    
    # Merge demands and time windows into coords
    # node 0 is depot
    nodes = []
    
    depot_row = coords[coords['ID'] == 0].iloc[0]
    nodes.append({
        'id': 0,
        'original_id': 0,
        'x': depot_row['X (km)'],
        'y': depot_row['Y (km)'],
        'demand_w': 0,
        'demand_v': 0,
        'tw_start': 0,
        'tw_end': 24,
        'is_green': False
    })
    
    for i in range(1, 99):
        c_row = coords[coords['ID'] == i].iloc[0]
        x, y = c_row['X (km)'], c_row['Y (km)']
        d_row = demands[demands['目标客户编号'] == i]
        dw = float(d_row['重量'].values[0]) if len(d_row) > 0 else 0.0
        dv = float(d_row['体积'].values[0]) if len(d_row) > 0 else 0.0
        
        t_row = tw[tw['客户编号'] == i]
        ts = float(t_row['start_h'].values[0]) if len(t_row) > 0 else 0.0
        te = float(t_row['end_h'].values[0]) if len(t_row) > 0 else 24.0
        
        # Check if in green zone (dist to center 0,0 <= 10)
        is_green = bool((x**2 + y**2)**0.5 <= 10.0)
        
        nodes.append({
            'id': i,
            'original_id': i,
            'x': x,
            'y': y,
            'demand_w': dw,
            'demand_v': dv,
            'tw_start': ts,
            'tw_end': te,
            'is_green': is_green
        })
            
    # 4. 距离矩阵
    dist_mat = pd.read_excel(base_dir + "距离矩阵.xlsx", header=None).values
    
    # Save to a format easy for the solver
    import json
    with open("问题1/processed_data.json", "w", encoding='utf-8') as f:
        json.dump({
            "nodes": nodes,
            "dist_mat": dist_mat.tolist()
        }, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    process_data()
    print("Data processing done.")
