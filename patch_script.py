import re

with open('fig/draw_anti_hub_real_data.py', 'r') as f:
    content = f.read()

old_str = """fig, axes = plt.subplots(2, 2, figsize=(16, 14))

# 【修改点1】不再提取半径导致的满屏黑点。而是抽样 Hub 与其最强的 36 个邻居，保持布局完全固定
if hub_name in G_raw:
    hub_edges = sorted(G_raw.edges(hub_name, data=True), key=lambda x: x[2]['weight'], reverse=True)
    sampled_neighbors = [v if u == hub_name else u for u, v, d in hub_edges[:35]]
    sampled_nodes = [hub_name] + sampled_neighbors
else:
    sampled_nodes = list(G_raw.nodes)[:36]

sub_raw = G_raw.subgraph(sampled_nodes)
sub_pruned = G_pruned.subgraph(sampled_nodes)
d_name = "Ginseng (HN1418)" if hub_name in ["人参", "HN1418"] else hub_name

# 锁定两图使用完全一样的坐标计算
pos = nx.spring_layout(sub_raw, seed=42, k=0.6)

# ---------- (a) 原始子图 (Hairball effect) ----------
ax1 = axes[0, 0]
nx.draw_networkx_nodes(sub_raw, pos, ax=ax1, nodelist=[hub_name], node_color='#D62728', node_size=450, edgecolors='black', linewidths=2)
nx.draw_networkx_nodes(sub_raw, pos, ax=ax1, nodelist=[n for n in sub_raw.nodes if n != hub_name], node_color='#CCCCCC', node_size=80, alpha=0.9, edgecolors='white')

# 控制透明薄边
weights_raw = [min(3.0, 0.5 + d['weight'] * 0.05) for u, v, d in sub_raw.edges(data=True)]
nx.draw_networkx_edges(sub_raw, pos, ax=ax1, alpha=0.25, edge_color='#888888', width=weights_raw)

ax1.set_title(f"(a) Raw Ego-Network Topology of '{d_name}'\\n(Over-dense Hairball effect)", fontsize=18, fontweight='bold', pad=15)
ax1.axis('off')

# ---------- (b) 剪枝后子图 (带有向箭头并弯曲处理) ----------
ax2 = axes[0, 1]
nx.draw_networkx_nodes(sub_pruned, pos, ax=ax2, nodelist=[hub_name] if hub_name in sub_pruned else [], node_color='#D62728', node_size=450, edgecolors='black', linewidths=2)
nx.draw_networkx_nodes(sub_pruned, pos, ax=ax2, nodelist=[n for n in sub_pruned.nodes if n != hub_name], node_color='#4C72B0', node_size=80, alpha=0.9, edgecolors='white')

edges_pruned = sub_pruned.edges(data=True)
weights_pruned = [min(2.5, 0.5 + d['weight'] * 5.0) for u, v, d in edges_pruned]

# 【关键】加入 arc3 让反向边不重叠叠黑，同时显示精准指向
nx.draw_networkx_edges(sub_pruned, pos, ax=ax2, alpha=0.7, 
                       edge_color='#2C3E50', width=weights_pruned, 
                       arrows=True, arrowsize=18, connectionstyle='arc3,rad=0.15')

ax2.set_title(f"(b) Pruned Asymmetric Ego-Network (Top-K={MAX_NEIGHBORS})\\n(Eq.2: Directed and Jaccard-driven)", fontsize=18, fontweight='bold', pad=15)
ax2.axis('off')

# ---------- (c) 修正为原始无向图的"共现频次"分布 (Classic Power-law Log-Log Scatter) ----------
ax3 = axes[1, 0]
from collections import Counter
# 使用完整的原始共现矩阵边权重 (`HH_cooc.data`) 来描绘绝对长尾特征
freq_counts = Counter(HH_cooc.data)
valid_freqs = {f: c for f, c in freq_counts.items() if f > 0}
x_freq = np.array(list(valid_freqs.keys()))
y_cnt = np.array(list(valid_freqs.values()))

# 使用散点图 (Scatter) 在双对数坐标系下展示教科书级别的向右下方倾斜直线
ax3.scatter(x_freq, y_cnt, color="#C44E52", alpha=0.8, edgecolor='black', linewidth=0.5, s=60)
ax3.set_xscale('log')
ax3.set_yscale('log')

ax3.set_title("(c) Co-occurrence Frequency Distribution", fontsize=18, fontweight='bold', pad=15)
ax3.set_xlabel("Co-occurrence Frequency between Herbs (Log Scale)", fontsize=14)
ax3.set_ylabel("Count of Herb Pairs (Log Scale)", fontsize=14)
ax3.grid(True, which="both", ls="--", alpha=0.2)

# 标记高光文本 (移至左下角或左上角空白处，避免遮挡数据)
ax3.text(0.35, 0.25, 'Strong Popularity Bias\\n(Classic Power-law)', transform=ax3.transAxes, 
         color='#D62728', fontsize=16, fontweight='bold', ha='center', bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray', boxstyle='round,pad=0.5'))

# ---------- (d) 剪枝后出度分布 (Bar Chart 精确离散) ----------
ax4 = axes[1, 1]
out_deg_counts = Counter([d for d in pruned_out_degrees if d > 0])
x_out = np.arange(1, MAX_NEIGHBORS + 3)
y_out = [out_deg_counts.get(i, 0) for i in x_out]

ax4.bar(x_out, y_out, color="#55A868", edgecolor="black", alpha=0.85, width=0.7)
ax4.set_title(f"(d) Subgraph Out-Degree Distribution (Max={MAX_NEIGHBORS})", fontsize=18, fontweight='bold', pad=15)
ax4.set_xlabel("Node Out-Degree Capacity Constraint", fontsize=14)
ax4.set_ylabel("Count of Herbs", fontsize=14)
ax4.set_xlim(0.5, MAX_NEIGHBORS + 1.5)
# 将刻度修改为逢2或逢1的整数，使其与柱子更对齐
ax4.set_xticks(range(1, MAX_NEIGHBORS + 2, max(1, MAX_NEIGHBORS // 5)))
ax4.grid(True, axis='y', ls="--", alpha=0.3)

ax4.text(0.2, 0.7, r'Information Bottleneck ($MAX\_K \leq 10$)', transform=ax4.transAxes, 
         color='#1B5E20', fontsize=16, fontweight='bold')

plt.tight_layout(pad=3.0)
save_path_pdf = os.path.join(CURRENT_DIR, "anti_hub_Ginseng_Optimized_V3.pdf")
save_path_png = os.path.join(CURRENT_DIR, "anti_hub_Ginseng_Optimized_V3.png")
plt.savefig(save_path_pdf, dpi=300, bbox_inches='tight')
plt.savefig(save_path_png, dpi=300, bbox_inches='tight')
print("✅ Fully Optimized Visualization V3 saved!")"""

new_str = """fig, axes = plt.subplots(2, 2, figsize=(16, 14))

# 取 Hub 节点与其最强的 35 个邻居，保持布局稳定
if hub_name in G_raw:
    hub_edges = sorted(G_raw.edges(hub_name, data=True), key=lambda x: x[2]['weight'], reverse=True)
    sampled_neighbors = [v if u == hub_name else u for u, v, d in hub_edges[:35]]
    sampled_nodes = [hub_name] + sampled_neighbors
else:
    sampled_nodes = list(G_raw.nodes)[:36]

sub_raw = G_raw.subgraph(sampled_nodes)
sub_pruned = G_pruned.subgraph(sampled_nodes)
d_name = "Ginseng" if hub_name in ["人参", "HN1418", "Ginseng"] else hub_name

# 锁定两图坐标计算一致
pos = nx.spring_layout(sub_raw, seed=42, k=0.6)

# 【配色系统统一 (Color Palette)】 - 遵循学术规范
COLOR_PRIM = '#4C72B0'     # 统一主色（节点/边/图柱）：深钢蓝
COLOR_TARGET = '#D62728'   # 目标锚点：红色，警示/突出
COLOR_RAW_EDGE = '#CCCCCC' # 原始边浅灰
COLOR_PRUN_EDGE = '#5975A4'# 剪枝后暗深灰蓝

# ---------- (a) 原始子图 ----------
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
ax2.axis('off')

# ---------- (c) 频次分布 (统一颜色、彻底去除高光文本框) ----------
ax3 = axes[1, 0]
from collections import Counter
freq_counts = Counter(HH_cooc.data)
valid_freqs = {f: c for f, c in freq_counts.items() if f > 0}
x_freq = np.array(list(valid_freqs.keys()))
y_cnt = np.array(list(valid_freqs.values()))

# 【修改：颜色统一为 COLOR_PRIM】
ax3.scatter(x_freq, y_cnt, color=COLOR_PRIM, alpha=0.7, edgecolor='black', linewidth=0.5, s=50)
ax3.set_xscale('log')
ax3.set_yscale('log')

ax3.set_title("(c) Co-occurrence Frequency Distribution", fontsize=18, fontweight='bold', pad=15)
ax3.set_xlabel("Co-occurrence Frequency (Log Scale)", fontsize=14)
ax3.set_ylabel("Count of Herb Pairs (Log Scale)", fontsize=14)
ax3.grid(True, which="both", ls="--", alpha=0.3)

# ---------- (d) 剪枝后出度分布 (精准对齐刻度，加入竖线替代文本) ----------
ax4 = axes[1, 1]
out_deg_counts = Counter([d for d in pruned_out_degrees if d > 0])

# 【修改：严格限制边界和连续整数刻度】
x_out = np.arange(1, MAX_NEIGHBORS + 1)
y_out = [out_deg_counts.get(i, 0) for i in x_out]

ax4.bar(x_out, y_out, color=COLOR_PRIM, edgecolor="black", alpha=0.85, width=0.7)
ax4.set_title("(d) Subgraph Out-Degree Distribution", fontsize=18, fontweight='bold', pad=15)
ax4.set_xlabel("Node Out-Degree", fontsize=14)
ax4.set_ylabel("Count of Herbs", fontsize=14)

# 确保无缝且对齐
ax4.set_xlim(0.5, MAX_NEIGHBORS + 0.5)
ax4.set_xticks(np.arange(1, MAX_NEIGHBORS + 1))
ax4.grid(True, axis='y', ls="--", alpha=0.3)

# 【修改：画优雅的垂直红虚线代替狗皮膏药】
ax4.axvline(x=MAX_NEIGHBORS, color=COLOR_TARGET, linestyle='--', linewidth=2, alpha=0.8)
ax4.text(MAX_NEIGHBORS - 0.2, max(y_out) * 0.85, f"Capacity K={MAX_NEIGHBORS}", 
         color=COLOR_TARGET, fontsize=14, fontweight='bold', ha='right')

plt.tight_layout(pad=3.0)
save_path_pdf = os.path.join(CURRENT_DIR, "anti_hub_Ginseng_Optimized_V4.pdf")
save_path_png = os.path.join(CURRENT_DIR, "anti_hub_Ginseng_Optimized_V4.png")
plt.savefig(save_path_pdf, dpi=300, bbox_inches='tight')
plt.savefig(save_path_png, dpi=300, bbox_inches='tight')
print("✅ Academic Refinement V4 saved!")"""

content = content.replace(old_str, new_str)
with open('fig/draw_anti_hub_real_data.py', 'w') as f:
    f.write(content)

