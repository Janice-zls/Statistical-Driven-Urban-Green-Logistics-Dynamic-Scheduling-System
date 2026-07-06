import json
import pandas as pd
import sys
sys.path.append('问题1')
from solve import load_data, VEHICLES, eval_edge, get_speed     

def main():
    nodes, dist_mat = load_data()
    with open('问题1/solution_opt.json', 'r', encoding='utf-8') as f:
        sol = json.load(f)

    vehicle_dict = {v['type']: v for v in VEHICLES}

    rows = []

    for idx, r in enumerate(sol['routes']):
        v_type = vehicle_dict[r['vehicle']]
        route = r['route']
        demands = r['demands']

        curr_w = sum(d['w'] for d in demands)
        t_curr = r.get('start_time', 8.0) # Start time

        timeline = []
        
        h_start = int(t_curr)
        m_start = int(round((t_curr - h_start) * 60))
        timeline.append(f"0({h_start:02d}:{m_start:02d})")

        for i in range(len(route) - 1):
            u = route[i]
            nxt = route[i+1]
            dist = dist_mat[u][nxt]

            load_ratio = curr_w / v_type['Q']
            travel_time, energy_cost = eval_edge(v_type, load_ratio, dist, t_curr)

            t_curr += travel_time

            if nxt != 0:
                dem = demands[i]
                node_info = nodes[dem['orig_id']]
                arrive_t = t_curr
                # Wait if arrive early
                t_curr = max(t_curr, node_info['tw_start'])
                # Service time
                t_curr += (20 / 60.0)
                curr_w -= dem['w']

                # Format time
                h = int(arrive_t)
                m = int(round((arrive_t - h) * 60))
                timeline.append(f"{nxt}({h:02d}:{m:02d})")
            else:
                h = int(t_curr)
                m = int(round((t_curr - h) * 60))
                timeline.append(f"0({h:02d}:{m:02d})")

        route_str = " -> ".join([str(n) for n in route])  
        time_str = " -> ".join(timeline)

        rows.append({
            '车辆编号': f"V{idx+1}",
            '车辆类型': v_type['type'],
            '固定成本(元)': v_type['cost'],
            '行驶与碳排放成本(元)': round(r['travel_cost'], 2),
            '时间窗惩罚成本(元)': round(r['penalty'], 2),
            '总成本(元)': round(r['cost'], 2),
            '行驶路径': route_str,
            '到达时间节点': time_str
        })

    df = pd.DataFrame(rows)
    df.to_excel('问题1/车辆调度方案.xlsx', index=False)
    print("Results saved to 问题1/车辆调度方案.xlsx")

if __name__ == '__main__':
    main()
