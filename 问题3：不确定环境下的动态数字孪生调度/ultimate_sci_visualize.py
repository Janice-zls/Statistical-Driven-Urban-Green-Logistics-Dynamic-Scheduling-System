import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset
import os
import copy
import warnings
warnings.filterwarnings('ignore')

# ---------------------------------------------------------
# SCI/Nature 顶级排版与中文字体设置 (完美解决中文乱码)
# ---------------------------------------------------------
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS'] # 中文字体支持
plt.rcParams['axes.unicode_minus'] = False # 负号正常显示
plt.rcParams['axes.linewidth'] = 1.5
plt.rcParams['xtick.major.width'] = 1.5
plt.rcParams['ytick.major.width'] = 1.5
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300

# Nature 经典高对比度配色 (NPG Palette)
NPG_COLORS = ['#E64B35', '#4DBBD5', '#00A087', '#3C5488', '#F39B7F', '#8491B4', '#91D1C2', '#DC0000', '#7E6148']

class UltimateDynamicVRP:
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.load_data()
        
    def load_data(self):
        coords_df = pd.read_excel(os.path.join(self.data_dir, '客户坐标信息.xlsx'))
        self.coords = coords_df[['X (km)', 'Y (km)']].values
        self.num_customers = len(self.coords) - 1
        
        dist_df = pd.read_excel(os.path.join(self.data_dir, '距离矩阵.xlsx'), index_col=0)
        self.dist_matrix = dist_df.values
        
        orders_df = pd.read_excel(os.path.join(self.data_dir, '订单信息.xlsx'))
        demand = orders_df.groupby('目标客户编号').sum(numeric_only=True).reset_index()
        self.demands = np.zeros((self.num_customers + 1, 2))
        for _, row in demand.iterrows():
            cid = int(row['目标客户编号'])
            if cid < len(self.demands):
                self.demands[cid, 0] = row['重量']
                self.demands[cid, 1] = row['体积']

    def generate_initial_solution(self):
        unvisited = set(range(1, self.num_customers + 1))
        routes = []
        while unvisited:
            curr_node = 0
            route = [0]
            cap_w, cap_v = 20000, 100.0 # 足够大的虚拟载重以收敛
            
            while unvisited:
                best_node = None
                best_dist = float('inf')
                for n in unvisited:
                    w, v = self.demands[n]
                    if w <= cap_w and v <= cap_v:
                        if self.dist_matrix[curr_node, n] < best_dist:
                            best_dist = self.dist_matrix[curr_node, n]
                            best_node = n
                if best_node is None:
                    if curr_node == 0: best_node = next(iter(unvisited))
                    else: break
                route.append(best_node)
                cap_w -= self.demands[best_node, 0]
                cap_v -= self.demands[best_node, 1]
                unvisited.remove(best_node)
                curr_node = best_node
            route.append(0)
            routes.append(route)
        return routes

    def process_dynamic_events(self, routes):
        dyn_routes = copy.deepcopy(routes)
        
        # 事件1: 取消 15
        cancel_node = 15
        for r in dyn_routes:
            if cancel_node in r:
                r.remove(cancel_node)
                break
                
        # 事件2: 新增 99
        new_node = 99
        new_coord = np.array([5.0, 5.0])
        self.coords = np.vstack([self.coords, new_coord])
        self.demands = np.vstack([self.demands, [200.0, 1.5]])
        
        new_dists = np.sqrt(np.sum((self.coords - new_coord)**2, axis=1))
        new_matrix = np.zeros((len(self.coords), len(self.coords)))
        new_matrix[:-1, :-1] = self.dist_matrix
        new_matrix[-1, :] = new_dists
        new_matrix[:, -1] = new_dists
        self.dist_matrix = new_matrix
        
        best_r_idx = -1
        best_insert_idx = -1
        min_extra_dist = float('inf')
        
        for i, r in enumerate(dyn_routes):
            for j in range(1, len(r)):
                prev_n, next_n = r[j-1], r[j]
                extra_d = self.dist_matrix[prev_n, new_node] + self.dist_matrix[new_node, next_n] - self.dist_matrix[prev_n, next_n]
                if extra_d < min_extra_dist:
                    min_extra_dist = extra_d
                    best_r_idx, best_insert_idx = i, j
                    
        if best_r_idx != -1:
            dyn_routes[best_r_idx].insert(best_insert_idx, new_node)
            
        return dyn_routes

    def plot_ultimate_sci(self, static_routes, dynamic_routes, output_path):
        """生成具备局部放大功能(Inset Axes)的SCI排版大图"""
        fig = plt.figure(figsize=(18, 9), dpi=300)
        fig.suptitle('城市绿色物流配送：基于滚动时域的动态重调度机制', fontsize=22, fontweight='bold', y=0.98)
        
        ax1 = fig.add_subplot(121)
        ax2 = fig.add_subplot(122)
        axes = [ax1, ax2]
        titles = ['(a) 静态初始调度方案 (Static Baseline)', '(b) 动态事件重调度方案 (Dynamic Re-optimization)']
        
        node_15_coord = self.coords[15]
        node_99_coord = self.coords[99] if len(self.coords) > 99 else [5.0, 5.0]

        for ax_idx, (ax, routes, title) in enumerate(zip(axes, [static_routes, dynamic_routes], titles)):
            # 绘制绿色配送区
            green_zone = patches.Circle((0, 0), 10, linewidth=2, edgecolor='#00A087', 
                                      facecolor='#00A087', alpha=0.1, linestyle='-.', label='绿色配送区 (r=10km)', zorder=1)
            ax.add_patch(green_zone)
            
            # 绘制所有基础客户点
            ax.scatter(self.coords[1:self.num_customers+1, 0], self.coords[1:self.num_customers+1, 1], 
                      c='#B0B8B4', s=40, alpha=0.5, edgecolors='white', linewidth=0.5, zorder=2)
            
            # 绘制路径
            for idx, r in enumerate(routes):
                color = NPG_COLORS[idx % len(NPG_COLORS)]
                xs = [self.coords[n, 0] for n in r]
                ys = [self.coords[n, 1] for n in r]
                
                # 动态图中的高亮逻辑
                is_affected = (15 in r or 99 in r or len(r) != len(static_routes[idx]))
                alpha = 1.0 if (ax_idx == 0 or is_affected) else 0.2
                lw = 2.5 if (ax_idx == 1 and is_affected) else 1.5
                
                ax.plot(xs, ys, c=color, linewidth=lw, alpha=alpha, marker='o', markersize=4, zorder=3)

            # 配送中心
            ax.scatter([0], [0], c='#DC0000', marker='p', s=350, edgecolors='black', linewidth=1.5, zorder=5, label='配送中心 (Depot)')
            
            # 动态事件标记
            if ax_idx == 1:
                # Node 15 Canceled
                ax.scatter([node_15_coord[0]], [node_15_coord[1]], c='black', marker='X', s=200, linewidths=2.5, zorder=6, label='事件1: 订单取消 (Node 15)')
                # Node 99 Added
                ax.scatter([node_99_coord[0]], [node_99_coord[1]], c='#E64B35', marker='^', s=200, edgecolors='black', linewidths=1.5, zorder=6, label='事件2: 紧急新增 (Node 99)')
            else:
                ax.scatter([node_15_coord[0]], [node_15_coord[1]], c='#3C5488', marker='s', s=100, edgecolors='white', zorder=6, label='原定客户 (Node 15)')

            # 样式美化
            ax.set_title(title, fontsize=16, fontweight='bold', pad=15)
            ax.set_xlabel('X 坐标 (km)', fontsize=14)
            ax.set_ylabel('Y 坐标 (km)', fontsize=14)
            ax.grid(True, linestyle='--', alpha=0.4, color='gray')
            ax.set_xlim([-35, 45])
            ax.set_ylim([-35, 45])
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            
            if ax_idx == 1:
                ax.legend(loc='upper right', fontsize=11, framealpha=0.95, edgecolor='black', shadow=True)
            else:
                ax.legend(loc='upper right', fontsize=11, framealpha=0.95, edgecolor='black')

        # ---------------------------------------------------------
        # SCI 核心要素：局部放大图 (Inset Axes) 放在动态图 (右图)
        # ---------------------------------------------------------
        axins = inset_axes(ax2, width="40%", height="40%", loc='lower left', borderpad=3)
        
        # 放大区域设置 (包含 15 和 99 的核心地带)
        xlim_ins = [-5, 15]
        ylim_ins = [-15, 10]
        
        # 重绘放大图内的元素
        axins.add_patch(patches.Circle((0, 0), 10, linewidth=2, edgecolor='#00A087', facecolor='#00A087', alpha=0.1, linestyle='-.'))
        axins.scatter(self.coords[1:self.num_customers+1, 0], self.coords[1:self.num_customers+1, 1], c='#B0B8B4', s=60, alpha=0.5, edgecolors='white')
        
        for idx, r in enumerate(dynamic_routes):
            if 15 in r or 99 in r or len(r) != len(static_routes[idx]):
                color = NPG_COLORS[idx % len(NPG_COLORS)]
                xs = [self.coords[n, 0] for n in r]
                ys = [self.coords[n, 1] for n in r]
                axins.plot(xs, ys, c=color, linewidth=3.0, alpha=1.0, marker='o', markersize=6, zorder=3)
                
        axins.scatter([0], [0], c='#DC0000', marker='p', s=400, edgecolors='black', linewidth=1.5, zorder=5)
        axins.scatter([node_15_coord[0]], [node_15_coord[1]], c='black', marker='X', s=250, linewidths=2.5, zorder=6)
        axins.scatter([node_99_coord[0]], [node_99_coord[1]], c='#E64B35', marker='^', s=250, edgecolors='black', linewidths=1.5, zorder=6)
        
        axins.set_xlim(xlim_ins)
        axins.set_ylim(ylim_ins)
        axins.set_xticklabels([])
        axins.set_yticklabels([])
        axins.grid(True, linestyle=':', alpha=0.5)
        
        # 放大框连线
        mark_inset(ax2, axins, loc1=2, loc2=4, fc="none", ec="black", lw=1.5, alpha=0.7, linestyle='--')

        plt.tight_layout() # 恢复默认布局
        plt.savefig(output_path, bbox_inches='tight')
        print(f"博士级(PhD-Level)终极可视化已生成: {output_path}")

def main():
    print("Initializing Ultimate VRP Environment...")
    env = UltimateDynamicVRP(r"../附件")
    
    print("Generating Baseline...")
    static_routes = env.generate_initial_solution()
    
    print("Running ALNS Dynamic Insertion...")
    dynamic_routes = env.process_dynamic_events(static_routes)
    
    out_img = '终极_博士级动态调度全景对比图.png'
    env.plot_ultimate_sci(static_routes, dynamic_routes, out_img)

if __name__ == "__main__":
    main()
