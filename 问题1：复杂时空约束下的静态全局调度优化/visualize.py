import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import seaborn as sns
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset

def setup_sci_style():
    # Nature/SCI Style aesthetics
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
        'axes.spines.top': False,
        'axes.spines.right': False,
        'axes.linewidth': 1.2,
        'xtick.major.width': 1.2,
        'ytick.major.width': 1.2,
        'font.size': 11,
        'axes.titlesize': 14,
        'axes.labelsize': 12,
        'legend.fontsize': 10,
        'legend.frameon': False,
        'figure.dpi': 300
    })

def visualize_results():
    setup_sci_style()
    
    # NPG Palette
    npg_colors = ['#E64B35', '#4DBBD5', '#00A087', '#3C5488', '#F39B7F', '#8491B4', '#91D1C2', '#DC0000']
    
    # Load data
    with open('问题1/processed_data.json', 'r', encoding='utf-8') as f:
        data = json.load(f)
    nodes = data['nodes']

    with open('问题1/solution.json', 'r', encoding='utf-8') as f:
        sol = json.load(f)
    routes = sol['routes']
    
    fig = plt.figure(figsize=(15, 8))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.3, 1], wspace=0.2)
    
    ax_map = fig.add_subplot(gs[0])
    ax_dist = fig.add_subplot(gs[1])
    
    # ==========================
    # Panel A: Spatial Route Map
    # ==========================
    # Green Zone
    gz = patches.Circle((0, 0), 10, facecolor='#00A087', alpha=0.1, 
                        edgecolor='#00A087', linestyle='--', linewidth=1.5, zorder=1)
    ax_map.add_patch(gz)
    
    # Prepare routes and colors
    v_types = [r['vehicle'] for r in routes]
    unique_v = sorted(list(set(v_types)))
    color_map = {v: npg_colors[i % len(npg_colors)] for i, v in enumerate(unique_v)}
    
    for r in routes:
        v_type = r['vehicle']
        route_nodes = r['route']
        x = [nodes[n]['x'] for n in route_nodes]
        y = [nodes[n]['y'] for n in route_nodes]
        ax_map.plot(x, y, color=color_map[v_type], alpha=0.35, linewidth=1.2, zorder=2)
        
    # Customers
    cx = [n['x'] for n in nodes if n['id'] != 0]
    cy = [n['y'] for n in nodes if n['id'] != 0]
    ax_map.scatter(cx, cy, s=15, c='#8491B4', alpha=0.6, edgecolors='none', zorder=3, label='Customers')
    
    # Depot
    dx, dy = nodes[0]['x'], nodes[0]['y']
    ax_map.scatter([dx], [dy], s=250, c='#DC0000', marker='*', edgecolors='black', linewidths=0.5, zorder=4, label='Depot')
    
    ax_map.set_xlabel('X Coordinate (km)', fontweight='bold')
    ax_map.set_ylabel('Y Coordinate (km)', fontweight='bold')
    ax_map.set_title('A  Spatial Distribution of Vehicle Routes', loc='left', fontweight='bold')
    ax_map.grid(True, linestyle=':', alpha=0.5)
    ax_map.set_aspect('equal', adjustable='datalim')
    
    # Custom Legend
    from matplotlib.lines import Line2D
    custom_lines = [Line2D([0], [0], color=color_map[v], lw=2, alpha=0.8) for v in unique_v]
    labels = [f'Vehicle Type {v} (Fuel)' if v in [1,2,3] else f'Vehicle Type {v} (Electric)' for v in unique_v]
    
    ax_map.legend(custom_lines + [ax_map.collections[0], ax_map.collections[1]], 
                  labels + ['Customers', 'Depot'],
                  loc='upper right', bbox_to_anchor=(1.05, 1.05))
                  
    # Inset Axes (Zoom on dense green zone)
    axins = inset_axes(ax_map, width="40%", height="40%", loc='lower left', 
                       bbox_to_anchor=(0.05, 0.05, 1, 1), bbox_transform=ax_map.transAxes)
    axins.add_patch(patches.Circle((0, 0), 10, facecolor='#00A087', alpha=0.1, 
                                   edgecolor='#00A087', linestyle='--', linewidth=1.5, zorder=1))
    
    for r in routes:
        v_type = r['vehicle']
        route_nodes = r['route']
        x = [nodes[n]['x'] for n in route_nodes]
        y = [nodes[n]['y'] for n in route_nodes]
        axins.plot(x, y, color=color_map[v_type], alpha=0.5, linewidth=1.0, zorder=2)
        
    axins.scatter(cx, cy, s=20, c='#8491B4', alpha=0.8, edgecolors='none', zorder=3)
    axins.scatter([0], [0], s=80, c='black', marker='+', zorder=4, label='City Center')
    
    # Focus on (-12, 12)
    axins.set_xlim(-12, 12)
    axins.set_ylim(-12, 12)
    axins.set_xticklabels([])
    axins.set_yticklabels([])
    axins.tick_params(bottom=False, left=False)
    for spine in axins.spines.values():
        spine.set_edgecolor('gray')
        spine.set_linewidth(1)
        spine.set_visible(True) # Inset needs full box
        
    mark_inset(ax_map, axins, loc1=2, loc2=4, fc="none", ec="0.5", alpha=0.5, linestyle='--')
    
    # ==========================
    # Panel B: Cost Distribution (Raincloud/Violin Plot)
    # ==========================
    df = pd.DataFrame(routes)
    df['Energy_Type'] = df['vehicle'].apply(lambda x: 'Electric' if x in [4, 5] else 'Fuel')
    df['vehicle_str'] = df['vehicle'].astype(str)
    
    # Violin Plot
    sns.violinplot(data=df, x='vehicle', y='cost', hue='Energy_Type', 
                   palette={'Fuel': '#E64B35', 'Electric': '#4DBBD5'}, 
                   inner="quart", alpha=0.6, ax=ax_dist, split=False)
                   
    # Strip Plot for data density (Raincloud element)
    sns.stripplot(data=df, x='vehicle', y='cost', color="black", alpha=0.4, 
                  jitter=True, size=4, ax=ax_dist)
                  
    ax_dist.set_xlabel('Vehicle Type', fontweight='bold')
    ax_dist.set_ylabel('Total Cost per Route (CNY)', fontweight='bold')
    ax_dist.set_title('B  Route Cost Distribution by Vehicle Type', loc='left', fontweight='bold')
    
    # Set y limits nicely
    ymin, ymax = ax_dist.get_ylim()
    ax_dist.set_ylim(max(0, ymin - 100), ymax * 1.1)
    
    # Adjust legend for subplot B
    handles, labels = ax_dist.get_legend_handles_labels()
    # sns violin might add duplicate legend handles, clean it up
    if len(handles) > 2:
        ax_dist.legend(handles[:2], labels[:2], title='Energy Type', loc='upper right')
    else:
        ax_dist.legend(title='Energy Type', loc='upper right')

    plt.tight_layout()
    
    # Save high quality figures
    plt.savefig('问题1/SCI级车辆调度路线图.png', dpi=300, bbox_inches='tight')
    plt.savefig('问题1/SCI级车辆调度路线图.pdf', bbox_inches='tight')
    print("Visualization saved successfully.")

if __name__ == '__main__':
    visualize_results()
