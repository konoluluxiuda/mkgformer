import os

with open('fig/draw_anti_hub_real_data.py', 'r', encoding='utf-8') as f:
    content = f.read()

old_a_b = """# ---------- (a) 原始子图 ----------
ax1 = axes[0, 0]
nx.draw_networkx_nodes(sub_raw, pos, ax=ax1, nodelist=[n for n in sub_raw.nodes if n != hub_name], node_color=COLOR_PRIM, node_size=80, alpha=0.5, edgecolors='white')
nx.draw_networkx_nodes(sub_raw, pos, ax=ax1, nodelist=[hub_name], node_color=COLOR_TARGET, node_size=450, edgecolors='black', linewidths=1.5)

weights_raw = [min(3.0, 0.5 + d['weight'] * 0.05) for u, v, d in sub_raw.edges(data=True)]
nx.draw_networkx_edges(sub_raw, pos, ax=ax1, alpha=0.25, edge_color=COLOR_RAW_EDGE, width=weights_raw)

ax1.set_title(f"(a) Raw Ego-Network of {d_name}", fontsize=18, fontweight='bold', pad=15)
ax1.axis('off')

# 【修改：添加统一图例 (Legend)】
import matplotlib.lines as mlines
legend_elements = [
    mlines.Line2D([0], [0], marker='o', color='w', label='Target Herb', markerfacecolor=COLOR_TARGET, markersize=12, markeredgecolor='k', markeredgewidth=1.5),
    mlines.Line2D([0], [0], marker='o', color='w', label='Neighbor Herbs', markerfacecolor=COLOR_PRIM, markersize=10, alpha=0.8),
    mlines.Line2D([0], [0], color=COLOR_PRUN_EDGE, lw=2, label='Directed Top-K Edge')
]
ax1.legend(handles=legend_elements, loc='upper left', fontsize=12, frameon=True, framealpha=0.9)

# ---------- (b) 剪枝后子图 (调小箭头并曲线化) ----------
ax2 = axes[0, 1]
nx.draw_networkx_nodes(sub_pruned, pos, ax=ax2, nodelist=[n for n in sub_pruned.nodes if n != hub_name], node_color=COLOR_PRIM, node_size=80, alpha=0.9, edgecolors='white')
nx.draw_networkx_nodes(sub_pruned, pos, ax=ax2, nodelist=[hub_name] if hub_name in sub_pruned else [], node_color=COLOR_TARGET, node_size=450, edgecolors='black', linewidths=1.5)

edges_pruned = sub_pruned.edges(data=True)
weights_pruned = [min(2.5, 0.5 + d['weight'] * 5.0) for u, v, d in edges_pruned]

# 【修改：减小 arrowsize，避免遮挡】
nx.draw_networkx_edges(sub_pruned, pos, ax=ax2, alpha=0.8, 
                       edge_color=COLOR_PRUN_EDGE, width=weights_pruned, 
                       arrows=True, arrowsize=12, connectionstyle='arc3,rad=0.2')

ax2.set_title(f"(b) Pruned Asymmetric Ego-Network", fontsize=18, fontweight='bold', pad=15)
ax2.axis('off')"""

new_a_b = """# ---------- (a) 原始子图 ----------
ax1 = axes[0, 0]
# 【修正】显式使用 sampled_nodes 所有节点迭代，确保前后视图的点集完全一致
nx.draw_networkx_nodes(sub_raw, pos, ax=ax1, nodelist=[n for n in sampled_nodes if n != hub_name], node_color=COLOR_PRIM, node_size=80, alpha=0.5, edgecolors='white')
nx.draw_networkx_nodes(sub_raw, pos, ax=ax1, nodelist=[hub_name], node_color=COLOR_TARGET, node_size=450, edgecolors='black', linewidths=1.5)

weights_raw = [min(3.0, 0.5 + d['weight'] * 0.05) for u, v, d in sub_raw.edges(data=True)]
nx.draw_networkx_edges(sub_raw, pos, ax=ax1, alpha=0.25, edge_color=COLOR_RAW_EDGE, width=weights_raw)

ax1.set_title(f"(a) Raw Ego-Network of {d_name}", fontsize=18, fontweight='bold', pad=15)
ax1.axis('off')

# 【修改：拆分图例，(a)图中只保留原始信息，并加入灰色连线的说明】
import matplotlib.lines as mlines
legend_elements_a = [
    mlines.Line2D([0], [0], marker='o', color='w', label='Target Herb', markerfacecolor=COLOR_TARGET, markersize=12, markeredgecolor='k', markeredgewidth=1.5),
    mlines.Line2D([0], [0], marker='o', color='w', label='Neighbor Herbs', markerfacecolor=COLOR_PRIM, markersize=10, alpha=0.8),
    mlines.Line2D([0], [0], color=COLOR_RAW_EDGE, lw=2, label='Raw Co-occurrence Edge', alpha=0.6)
]
ax1.legend(handles=legend_elements_a, loc='upper left', fontsize=12, frameon=True, framealpha=0.9)

# ---------- (b) 剪枝后子图 (调小箭头并曲线化) ----------
ax2 = axes[0, 1]
# 【修正】即使修剪后 hub 的边数量发生了剧变，也强制把红色中心点和原本的邻居框架画出来，保持视觉对齐
nx.draw_networkx_nodes(sub_raw, pos, ax=ax2, nodelist=[n for n in sampled_nodes if n != hub_name], node_color=COLOR_PRIM, node_size=80, alpha=0.9, edgecolors='white')
nx.draw_networkx_nodes(sub_raw, pos, ax=ax2, nodelist=[hub_name], node_color=COLOR_TARGET, node_size=450, edgecolors='black', linewidths=1.5)

edges_pruned = sub_pruned.edges(data=True)
# 【修正】将修剪后的目标有向边宽度变细
weights_pruned = [min(1.5, 0.5 + d['weight'] * 2.0) for u, v, d in edges_pruned]

# 【修正】将箭头大小减半 (arrowsize 从 12 降至 7)，大幅减少拥挤感
nx.draw_networkx_edges(sub_pruned, pos, ax=ax2, alpha=0.8, 
                       edge_color=COLOR_PRUN_EDGE, width=weights_pruned, 
                       arrows=True, arrowsize=7, connectionstyle='arc3,rad=0.2')

ax2.set_title(f"(b) Pruned Asymmetric Ego-Network", fontsize=18, fontweight='bold', pad=15)
ax2.axis('off')

# 【修改：在(b)图中补上专属于修剪后的箭头图例】
legend_elements_b = [
    mlines.Line2D([0], [0], color=COLOR_PRUN_EDGE, lw=2, label='Directed Top-K Edge')
]
ax2.legend(handles=legend_elements_b, loc='upper left', fontsize=12, frameon=True, framealpha=0.9)"""

content = content.replace(old_a_b, new_a_b)

old_c = """ax3.set_xlabel("Co-occurrence Frequency (Log Scale)", fontsize=14)
ax3.set_ylabel("Count of Herb Pairs (Log Scale)", fontsize=14)"""
new_c = """ax3.set_xlabel("Co-occurrence Frequency", fontsize=14)
ax3.set_ylabel("Count of Herb Pairs", fontsize=14)"""
content = content.replace(old_c, new_c)

old_d = """ax4.text(MAX_NEIGHBORS - 0.2, max(y_out) * 0.85, f"Capacity K={MAX_NEIGHBORS}", 
         color=COLOR_TARGET, fontsize=14, fontweight='bold', ha='right')"""
new_d = """ax4.text(MAX_NEIGHBORS - 0.4, max(y_out) * 0.90, f"Capacity K={MAX_NEIGHBORS}", 
         color=COLOR_TARGET, fontsize=14, ha='right')"""
content = content.replace(old_d, new_d)

with open('fig/draw_anti_hub_real_data.py', 'w', encoding='utf-8') as f:
    f.write(content)

