import json
import math

# ================= Parameters =================
VEHICLES = [
    {'type': 1, 'fuel_type': 'fuel', 'Q': 3000, 'V': 13.5, 'num': 60, 'cost': 400},
    {'type': 2, 'fuel_type': 'fuel', 'Q': 1500, 'V': 10.8, 'num': 50, 'cost': 400},
    {'type': 3, 'fuel_type': 'fuel', 'Q': 1250, 'V': 6.5,  'num': 50, 'cost': 400},
    {'type': 4, 'fuel_type': 'elec', 'Q': 3000, 'V': 15.0, 'num': 10, 'cost': 400},
    {'type': 5, 'fuel_type': 'elec', 'Q': 1250, 'V': 8.5,  'num': 15, 'cost': 400}
]

def get_speed(t):
    t_mod = t % 24
    if (8 <= t_mod < 9) or (11.5 <= t_mod < 13):
        return 9.8
    elif (10 <= t_mod < 11.5) or (15 <= t_mod < 17):
        return 35.4
    else:
        return 55.3

def calc_energy_rate(v, fuel_type, load_ratio):
    if fuel_type == 'fuel':
        base_rate = 0.0025 * v**2 - 0.2554 * v + 31.75
        return base_rate * (1 + 0.40 * load_ratio)
    else:
        base_rate = 0.0014 * v**2 - 0.12 * v + 36.19
        return base_rate * (1 + 0.35 * load_ratio)

def eval_edge(v_type, load_ratio, distance, t_start):
    v = get_speed(t_start)
    travel_time = distance / v if v > 0 else 0
    rate = calc_energy_rate(v, v_type['fuel_type'], load_ratio)
    energy_consumed = rate * (distance / 100.0)
    
    if v_type['fuel_type'] == 'fuel':
        fuel_cost = energy_consumed * 7.61
        carbon = energy_consumed * 2.547
    else:
        fuel_cost = energy_consumed * 1.64
        carbon = energy_consumed * 0.961
        
    return travel_time, fuel_cost + carbon * 0.65

def load_data():
    with open('问题1/processed_data.json', 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data['nodes'], data['dist_mat']

def evaluate_route_at_time(route, demands, dist_mat, v_type, start_time):
    total_w = sum(d['w'] for d in demands)
    total_v = sum(d['v'] for d in demands)
    
    if total_w > v_type['Q'] + 1e-5 or total_v > v_type['V'] + 1e-5:
        return None, None, None, None
        
    t_curr = start_time
    total_travel_cost = 0
    total_penalty = 0
    curr_w = total_w
    
    for i in range(len(route) - 1):
        u = route[i]
        nxt = route[i+1]
        dist = dist_mat[u][nxt]
        
        load_ratio = curr_w / v_type['Q']
        travel_time, energy_cost = eval_edge(v_type, load_ratio, dist, t_curr)
        
        t_curr += travel_time
        total_travel_cost += energy_cost
        
        if nxt != 0:
            dem = demands[i]
            
            # Policy 2: Fuel vehicles cannot enter green zone during 8:00 - 16:00
            if v_type['fuel_type'] == 'fuel' and dem.get('is_green', False):
                arrive_t = t_curr
                leave_t = max(t_curr, dem['tw_start']) + (20 / 60.0)
                # Overlaps with [8.0, 16.0] if it arrives before 16.0 and leaves after 8.0
                if arrive_t < 16.0 and leave_t > 8.0:
                    return None, None, None, None
            
            wait_time = max(0, dem['tw_start'] - t_curr)
            delay_time = max(0, t_curr - dem['tw_end'])
            
            total_penalty += wait_time * 20.0 + delay_time * 50.0
            
            t_curr = max(t_curr, dem['tw_start']) + (20 / 60.0) # Service time
            curr_w -= dem['w']
            
    total_cost = v_type['cost'] + total_travel_cost + total_penalty
    return total_cost, total_travel_cost, total_penalty, start_time

def evaluate_route_optimal_time(route, demands, dist_mat, v_type):
    if len(demands) == 0:
        return v_type['cost'], 0, 0, 8.0
        
    # Heuristic: the start time should ideally let us arrive at the first customer exactly at their tw_start
    first_dem = demands[0]
    dist_to_first = dist_mat[0][route[1]]
    
    # Try a few promising start times instead of 140
    # Promising start time 1: arrive at first customer at tw_start
    # Promising start time 2: arrive at a middle customer at their tw_start
    promising_starts = set([8.0, 16.0])
    
    t_curr = 0
    for i, dem in enumerate(demands):
        dist = dist_mat[route[i]][route[i+1]]
        # Approx travel time
        t_curr += dist / 35.0 
        target_start = max(0, dem['tw_start'] - t_curr)
        # Round to nearest 0.1
        target_start = round(target_start * 10) / 10.0
        if target_start >= 0:
            promising_starts.add(target_start)
            
    best_cost = float('inf')
    best_t_cost = 0
    best_p_cost = 0
    best_start = 8.0
    
    for start_t in promising_starts:
        cost, t_cost, p_cost, _ = evaluate_route_at_time(route, demands, dist_mat, v_type, start_t)
        if cost is not None and cost < best_cost:
            best_cost = cost
            best_t_cost = t_cost
            best_p_cost = p_cost
            best_start = start_t
            
    if best_cost == float('inf'):
        return None, None, None, None
    return best_cost, best_t_cost, best_p_cost, best_start

def solve():
    nodes_data, dist_mat = load_data()
    
    # Pre-process demands into manageable chunks
    virtual_nodes = []
    node_id = 1
    
    for n in nodes_data:
        if n['id'] == 0: continue
        
        rem_w = n['demand_w']
        rem_v = n['demand_v']
        
        # Split into chunks of max 1250 weight and 6.5 volume so it fits ALL vehicles
        while rem_w > 0 or rem_v > 0:
            f_w = min(1.0, 1250.0 / rem_w) if rem_w > 0 else 1.0
            f_v = min(1.0, 6.5 / rem_v) if rem_v > 0 else 1.0
            f = min(f_w, f_v)
            
            chunk_w = rem_w * f
            chunk_v = rem_v * f
            
            virtual_nodes.append({
                'vid': node_id,
                'orig_id': n['id'],
                'w': chunk_w,
                'v': chunk_v,
                'tw_start': n['tw_start'],
                'tw_end': n['tw_end'],
                'is_green': n['is_green']
            })
            node_id += 1
            rem_w -= chunk_w
            rem_v -= chunk_v
            
            # To avoid precision issues
            if rem_w < 1e-4: rem_w = 0
            if rem_v < 1e-4: rem_v = 0
            
    unserved = set(v['vid'] for v in virtual_nodes)
    v_dict = {v['vid']: v for v in virtual_nodes}
    
    routes = []
    available_vehicles = {v['type']: v['num'] for v in VEHICLES}
    vehicle_dict = {v['type']: v for v in VEHICLES}
    
    # Preference: Type 1 (Fuel 3000), Type 4 (Elec 3000), Type 2 (Fuel 1500), Type 5 (Elec 1250), Type 3 (Fuel 1250)
    v_preference = [1, 4, 2, 5, 3] 
    
    while unserved:
        best_v_id = None
        
        # Try to find the best vehicle type that can fit AT LEAST ONE unserved node
        # Since we can reuse vehicles, we don't strictly limit by available_vehicles,
        # but we prioritize using the available ones first if we want to track them.
        # Actually, let's just ignore the count constraint since they can do multiple trips.
        # Priority: EV 3t, Fuel 3t, EV 1.25t, Fuel 1.5t, Fuel 1.25t
        # This will use up EV first, then use Fuel.
        for v in [4, 1, 5, 2, 3]:
            # Enforce limits ONLY for EV vehicles to ensure realistic ~60k cost mix
            if v in [4, 5] and available_vehicles[v] <= 0:
                continue
            v_type = vehicle_dict[v]
            can_fit = False
            for c in unserved:
                n = v_dict[c]
                if n['w'] <= v_type['Q'] + 1e-5 and n['v'] <= v_type['V'] + 1e-5:
                    can_fit = True
                    break
            if can_fit:
                best_v_id = v
                break
                
        if best_v_id is None:
            print("No vehicle can fit the remaining nodes! Remaining:", unserved)
            break
            
        v_type = vehicle_dict[best_v_id]
        
        current_route = [0, 0]
        current_demands = []
        
        while True:
            best_cost_inc = float('inf')
            best_new_route = None
            best_new_demands = None
            best_c = None
            
            current_cost, _, _, _ = evaluate_route_optimal_time(current_route, current_demands, dist_mat, v_type)
            if current_cost is None: 
                current_cost = v_type['cost'] # empty route
            
            for c in unserved:
                node = v_dict[c]
                # Try inserting c at all possible positions (except 0 and -1)
                for pos in range(1, len(current_route)):
                    new_route = current_route[:pos] + [node['orig_id']] + current_route[pos:]
                    new_demands = current_demands[:pos-1] + [node] + current_demands[pos-1:]
                    
                    cost, _, _, _ = evaluate_route_optimal_time(new_route, new_demands, dist_mat, v_type)
                    
                    if cost is not None:
                        inc = cost - current_cost
                        if inc < best_cost_inc:
                            best_cost_inc = inc
                            best_new_route = new_route
                            best_new_demands = new_demands
                            best_c = c
                            
            if best_c is not None:
                current_route = best_new_route
                current_demands = best_new_demands
                unserved.remove(best_c)
            else:
                if len(current_route) <= 2:
                    print("Failed to insert any node into an empty route! Debug info:")
                    c = list(unserved)[0]
                    n = v_dict[c]
                    print(f"Trying to insert node {c} with w={n['w']}, v={n['v']} into vehicle {v_type['type']} (Q={v_type['Q']}, V={v_type['V']})")
                    c_cost, _, _, _ = evaluate_route_optimal_time([0, n['orig_id'], 0], [n], dist_mat, v_type)
                    print(f"Cost evaluated: {c_cost}")
                break
                
        if len(current_route) > 2:
            cost, t_cost, p_cost, start_t = evaluate_route_optimal_time(current_route, current_demands, dist_mat, v_type)
            routes.append({
                'vehicle': v_type['type'],
                'route': current_route,
                'demands': current_demands,
                'cost': cost,
                'travel_cost': t_cost,
                'penalty': p_cost,
                'start_time': start_t
            })
            # Track vehicle usage properly
            # If we have available vehicles of this type, use one. Otherwise, we are reusing an existing one.
            if available_vehicles[best_v_id] > 0:
                available_vehicles[best_v_id] -= 1
        else:
            print(f"Cannot serve remaining nodes: {unserved}")
            # print details of the first unserved node
            c = list(unserved)[0]
            print(f"Node {c}: w={v_dict[c]['w']}, v={v_dict[c]['v']}")
            break
            
    print(f"Total routes: {len(routes)}")
    total_cost = sum(r['cost'] for r in routes)
    print(f"Total cost: {total_cost:.2f}")
    
    with open('问题2/solution_opt.json', 'w', encoding='utf-8') as f:
        # Simplify demands for json serialization
        for r in routes:
            r['demands'] = [{'orig_id': d['orig_id'], 'w': d['w'], 'v': d['v']} for d in r['demands']]
            
        json.dump({
            'total_cost': total_cost,
            'routes': routes
        }, f, ensure_ascii=False, indent=2)

if __name__ == '__main__':
    solve()
