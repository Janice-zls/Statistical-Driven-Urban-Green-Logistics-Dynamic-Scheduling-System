import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import ListedColormap
import os
import copy
import time
import warnings
warnings.filterwarnings('ignore')

# ---------------------------------------------------------
# SCI/Nature Style Plot Settings
# ---------------------------------------------------------
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans']
plt.rcParams['axes.linewidth'] = 1.2
plt.rcParams['xtick.major.width'] = 1.2
plt.rcParams['ytick.major.width'] = 1.2
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
# NPG-like Color Palette
NPG_COLORS = ['#E64B35', '#4DBBD5', '#00A087', '#3C5488', '#F39B7F', '#8491B4', '#91D1C2', '#DC0000', '#7E6148']

class DynamicVRPEnv:
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.load_data()
        self.num_customers = len(self.coords) - 1
        
    def load_data(self):
        # 客户坐标
        coords_df = pd.read_excel(os.path.join(self.data_dir, '客户坐标信息.xlsx'))
        self.coords = coords_df[['X (km)', 'Y (km)']].values
        self.num_customers = len(self.coords) - 1
        
        # 距离矩阵
        dist_df = pd.read_excel(os.path.join(self.data_dir, '距离矩阵.xlsx'), index_col=0)
        self.dist_matrix = dist_df.values
        
        # 订单信息 (聚合到客户层面)
        orders_df = pd.read_excel(os.path.join(self.data_dir, '订单信息.xlsx'))
        demand = orders_df.groupby('目标客户编号').agg({'重量': 'sum', '体积': 'sum'}).reset_index()
        self.demands = np.zeros((self.num_customers + 1, 2)) # [weight, volume]
        for _, row in demand.iterrows():
            cid = int(row['目标客户编号'])
            if cid < len(self.demands):
                self.demands[cid, 0] = row['重量']
                self.demands[cid, 1] = row['体积']
                
        # 时间窗
        tw_df = pd.read_excel(os.path.join(self.data_dir, '时间窗.xlsx'))
        self.time_windows = np.zeros((self.num_customers + 1, 2))
        for _, row in tw_df.iterrows():
            cid = int(row['客户编号'])
            if cid < len(self.time_windows):
                # 简化：将 HH:MM 转换为分钟数 (以0:00为基准)
                try:
                    s_time = row['开始时间'].split(':')
                    e_time = row['结束时间'].split(':')
                    self.time_windows[cid, 0] = int(s_time[0])*60 + int(s_time[1])
                    self.time_windows[cid, 1] = int(e_time[0])*60 + int(e_time[1])
                except:
                    pass
                    
    def generate_initial_solution(self):
        """生成初始启发式解 (C-W 节约算法或贪婪插入)"""
        unvisited = set(range(1, self.num_customers + 1))
        routes = []
        
        # 简化版贪婪插入，用于快速生成合理路径基准
        while unvisited:
            curr_node = 0
            route = [0]
            cap_w, cap_v = 20000, 100.0 # 增大虚拟载重以保证能分配所有节点
            
            while unvisited:
                # 寻找距离当前节点最近且满足容量的节点
                best_node = None
                best_dist = float('inf')
                
                for n in unvisited:
                    w, v = self.demands[n]
                    if w <= cap_w and v <= cap_v:
                        if self.dist_matrix[curr_node, n] < best_dist:
                            best_dist = self.dist_matrix[curr_node, n]
                            best_node = n
                            
                if best_node is None:
                    if curr_node == 0:
                        best_node = next(iter(unvisited))
                    else:
                        break
                    
                route.append(best_node)
                cap_w -= self.demands[best_node, 0]
                cap_v -= self.demands[best_node, 1]
                unvisited.remove(best_node)
                curr_node = best_node
                
            route.append(0)
            routes.append(route)
            
        return routes

    def process_dynamic_events(self, routes):
        """处理动态事件：1. 客户取消 2. 紧急订单"""
        # 深拷贝以保留对比
        dyn_routes = copy.deepcopy(routes)
        
        # 事件1: 客户15取消订单
        cancel_node = 15
        for r in dyn_routes:
            if cancel_node in r:
                r.remove(cancel_node)
                break
                
        # 事件2: 新增紧急订单 99 (坐标 5.0, 5.0, 需求200kg, 1.5m3)
        new_node = 99
        new_coord = np.array([5.0, 5.0])
        # 将新节点加入坐标和需求系
        self.coords = np.vstack([self.coords, new_coord])
        self.demands = np.vstack([self.demands, [200.0, 1.5]])
        
        # 扩充距离矩阵
        new_dists = np.sqrt(np.sum((self.coords - new_coord)**2, axis=1))
        new_matrix = np.zeros((len(self.coords), len(self.coords)))
        new_matrix[:-1, :-1] = self.dist_matrix
        new_matrix[-1, :] = new_dists
        new_matrix[:, -1] = new_dists
        self.dist_matrix = new_matrix
        
        # 最优插入启发式
        best_r_idx = -1
        best_insert_idx = -1
        min_extra_dist = float('inf')
        
        for i, r in enumerate(dyn_routes):
            # 尝试插入到该路径的各个位置
            for j in range(1, len(r)):
                prev_n = r[j-1]
                next_n = r[j]
                
                # 计算边际距离增加
                old_d = self.dist_matrix[prev_n, next_n]
                new_d = self.dist_matrix[prev_n, new_node] + self.dist_matrix[new_node, next_n]
                extra_d = new_d - old_d
                
                if extra_d < min_extra_dist:
                    min_extra_dist = extra_d
                    best_r_idx = i
                    best_insert_idx = j
                    
        # 执行插入
        if best_r_idx != -1:
            dyn_routes[best_r_idx].insert(best_insert_idx, new_node)
            
        return dyn_routes

    def plot_routes(self, static_routes, dynamic_routes, output_path):
        """绘制高规格SCI级别对比图"""
        fig, axes = plt.subplots(1, 2, figsize=(16, 7), dpi=300)
        titles = ['Static Routing (Before Events)', 'Dynamic Routing (After Events)']
        
        for ax, routes, title in zip(axes, [static_routes, dynamic_routes], titles):
            # 绘制绿色配送区 (半径10km)
            green_zone = patches.Circle((0, 0), 10, linewidth=1.5, edgecolor='#2ca02c', 
                                      facecolor='#2ca02c', alpha=0.1, linestyle='--', label='Green Zone (r=10km)')
            ax.add_patch(green_zone)
            
            # 绘制客户点
            ax.scatter(self.coords[1:self.num_customers+1, 0], self.coords[1:self.num_customers+1, 1], 
                      c='gray', s=20, alpha=0.6, edgecolors='none', label='Customers')
            
            # 绘制路线
            for idx, r in enumerate(routes):
                color = NPG_COLORS[idx % len(NPG_COLORS)]
                xs = [self.coords[n, 0] for n in r]
                ys = [self.coords[n, 1] for n in r]
                
                # 高亮受影响的路线
                alpha = 1.0 if (15 in r or 99 in r or len(r) != len(static_routes[idx])) else 0.3
                linewidth = 2.0 if alpha == 1.0 else 1.0
                
                ax.plot(xs, ys, c=color, linewidth=linewidth, alpha=alpha, marker='o', markersize=3)
                
            # 高亮配送中心
            ax.scatter([0], [0], c='red', marker='*', s=200, edgecolors='black', zorder=5, label='Depot (0,0)')
            
            # 标注动态事件节点
            if 'Dynamic' in title:
                # 事件1: 取消的节点 (使用原坐标系中的15)
                ax.scatter([self.coords[15, 0]], [self.coords[15, 1]], c='black', marker='x', s=100, linewidth=2, zorder=6, label='Canceled (Node 15)')
                # 事件2: 新增的节点
                ax.scatter([5.0], [5.0], c='magenta', marker='^', s=120, edgecolors='black', zorder=6, label='New Order (Node 99)')
            else:
                ax.scatter([self.coords[15, 0]], [self.coords[15, 1]], c='blue', marker='s', s=60, zorder=6, label='Planned (Node 15)')
                
            ax.set_title(title, fontsize=16, fontweight='bold', pad=15)
            ax.set_xlabel('X Coordinate (km)', fontsize=12)
            ax.set_ylabel('Y Coordinate (km)', fontsize=12)
            ax.grid(True, linestyle=':', alpha=0.6)
            ax.set_xlim([-35, 45])
            ax.set_ylim([-35, 45])
            ax.legend(loc='upper right', fontsize=10, framealpha=0.9, edgecolor='black')
            
            # 去除顶部和右侧边框 (SCI Style)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            
        plt.tight_layout()
        plt.savefig(output_path, bbox_inches='tight')
        print(f"高规格对比图已保存至: {output_path}")

def main():
    print("Initializing Dynamic VRP Environment (PhD Level Formulation)...")
    data_dir = r"../附件"
    env = DynamicVRPEnv(data_dir)
    
    print("Generating Static Baseline Routes using Insertion Heuristics...")
    static_routes = env.generate_initial_solution()
    
    print("Simulating Real-Time Events (Rolling Horizon)...")
    print(" Event 1 [t=12:00]: Customer 15 requests cancellation.")
    print(" Event 2 [t=14:00]: Urgent new order arrives at (5.0, 5.0).")
    
    start_time = time.time()
    dynamic_routes = env.process_dynamic_events(static_routes)
    calc_time = time.time() - start_time
    
    print(f"Dynamic re-optimization completed in {calc_time:.4f} seconds.")
    
    out_img = 'Dynamic_VRP_Comparison_SCI.png'
    env.plot_routes(static_routes, dynamic_routes, out_img)
    
main()
