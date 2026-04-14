import os
import random
import pandas as pd
import numpy as np
from scipy import sparse
import matplotlib.pyplot as plt
import networkx as nx

# =========================================================
# 1. 样式配置
# =========================================================
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman', 'DejaVu Serif']
plt.rcParams['axes.unicode_minus'] = False 

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_ROOT = os.path.join(CURRENT_DIR, '..', 'MKG', 'dataset', 'NEWHERB', 'kge_data')

herb_disease_pairs = []
for f in ['train.tsv', 'dev.tsv', 'test.tsv']:
    path = os.path.join(DATA_ROOT, f)
    if os.path.exists(path):
        df = pd.read_csv(path, sep='\t', header=None, names=['h', 'r', 't'])
        hd = df[df['r'] == 'treats_disease']
        for _, row in hd.iterrows():
            herb_disease_pairs.append((row['h'], row['t']))

herbs = list(set([p[0] for p in herb_disease_pairs]))
diseases = list(set([p[1] for p in herb_disease_pairs]))
h_idx = {h: i for i, h in enumerate(herbs)}
d_idx = {d: i for i, d in enumerate(diseases)}

# =========================================================
# 2. 构建 Herb-Disease 矩阵 & Jaccard 相似度计算 (Eq.1)
# =========================================================
rows = [h_idx[p[0]] for p in herb_disease_pairs]
cols = [d_idx[p[1]] for p in herb_disease_pairs]
vals = [1] * len(rows)

H_D_mat = sparse.csr_matrix((vals, (rows, cols)), shape=(len(herbs), len(diseases)))
HH_cooc = H_D_mat @ H_D_mat.T
HH_cooc.setdiag(0)

herb_degrees = np.array(H_D_mat.sum(axis=1)).flatten()
inter_dense = HH_cooc.toarray()
num_herbs = len(herbs)

jaccard_mat = np.zeros((num_herbs, num_herbs))
for i in range(num_herbs):
    for j in range(num_herbs):
        if inter_dense[i, j] > 0:
            union = herb_degrees[i] + herb_degrees[j] - inter_dense[i, j]
            jaccard_mat[i, j] = inter_dense[i, j] / union

# =========================================================
# 3. 模拟长尾去噪 / Top-K Jaccard 截断 (Eq.2)
# =========================================================
MAX_NEIGHBORS = 10
MIN_SIM = 0.8
MIN_FREQ = 1

raw_node_degrees = []
G_raw = nx.Graph()
G_pruned = nx.DiGraph()

for i in range(len(herbs)):
    row_cooc = HH_cooc.getrow(i)
    raw_node_degrees.append(len(row_cooc.indices))
    
    for j, val in zip(row_cooc.indices, row_cooc.data):
        if i < j and val >= MIN_FREQ: G_raw.add_edge(herbs[i], herbs[j], weight=val)
    
    jaccard_row = jaccard_mat[i]
    valid_indices = np.where(jaccard_row >= MIN_SIM)[0]
    valid_neighbors = [(j, jaccard_row[j]) for j in valid_indices if j != i]
    
    valid_neighbors.sort(key=lambda x: x[1], reverse=True)
    pruned_neighbors = valid_neighbors[:MAX_NEIGHBORS]
    
    for j, val in pruned_neighbors:
        G_pruned.add_edge(herbs[i], herbs[j], weight=val)

pruned_out_degrees = [d for n, d in G_pruned.out_degree()] 

# =========================================================
# 4. 指定 Hub 节点并完美布局画图
# =========================================================
import matplotlib.lines as mlines

TARGET_NAMES = ["人参", "HN1418", "Ginseng"]
if raw_node_degrees:
    hub_name = next((name for name in TARGET_NAMES if name in herbs), herbs[np.argmax(raw_node_degrees)])
else:
    hub_name = herbs[0] if herbs else "Ginseng"

fig, axes = plt.subplots(2, 2, figsize=(16, 14))

# 提取 Hub 与其最强的 35 个邻居，真实反映原始图拓扑结构
if hub_name in G_raw:
    hub_edges = sorted(G_raw.edges(hub_name, data=True), key=lambda x: x[2]['weight'], reverse=True)
    sampled_neighbors = [v if u == hub_name else u for u, v, d in hub_edges[:35]]
    sampled_nodes = [hub_name] + sampled_neighbors
else:
    sampled_nodes = list(G_raw.nodes)[:36]

sub_raw = G_raw.subgraph(sampled_nodes)
sub_pruned = G_pruned.subgraph(sampled_nodes)
d_name = "Ginseng (HN1418)" if hub_name in ["人参", "HN1418"] else hub_name

# 【终极恢复修复】用圆环分布初始化，保证原图和剪枝图完全锚定，杜绝聚集和太空垃圾失控
initial_pos = {hub_name: (0, 0)}
angle = 0
step = 2 * np.pi / (len(sampled_nodes) - 1)
for n in sampled_nodes:
    if n != hub_name:
        initial_pos[n] = (np.cos(angle), np.sin(angle))
        angle += step

# 基于原始高密度连线的子图来控制引力，保证呈现真正的 Ego-Network 外放感
pos = nx.spring_layout(sub_raw, seed=1024, k=0.5, pos=initial_pos, fixed=[hub_name])

# 统一高级配色
COLOR_HUB = '#D62728'     
COLOR_NODE = '#4C72B0'    
COLOR_EDGE_RAW = '#CCCCCC'
COLOR_EDGE_PRUNED = '#4C72B0'

# ---------- (a) 原始子图 ----------
ax1 = axes[0, 0]
nx.draw_networkx_nodes(sub_raw, pos, ax=ax1, nodelist=[hub_name], node_color=COLOR_HUB, node_size=600, edgecolors='black', linewidths=2.5)
nx.draw_networkx_nodes(sub_raw, pos, ax=ax1, nodelist=[n for n in sub_raw.nodes if n != hub_name], node_color=COLOR_NODE, node_size=100, alpha=0.7, edgecolors='white', linewidths=1.5)
weights_raw = [min(3.0, 0.5 + d['weight'] * 0.05) for u, v, d in sub_raw.edges(data=True)]
nx.draw_networkx_edges(sub_raw, pos, ax=ax1, alpha=0.25, edge_color=COLOR_EDGE_RAW, width=weights_raw)
ax1.set_title(f"(a) Raw Ego-Network of {d_name}", fontsize=18, fontweight='bold', pad=15)
ax1.axis('off')

# ---------- (b) 剪枝后子图 ----------
ax2 = axes[0, 1]
nx.draw_networkx_nodes(sub_pruned, pos, ax=ax2, nodelist=[hub_name] if hub_name in sub_pruned else [], node_color=COLOR_HUB, node_size=600, edgecolors='black', linewidths=2.5)

# 过滤掉在剪枝图中完全没有连线的孤立节点
active_nodes = [n for n in sub_pruned.nodes if n != hub_name and sub_pruned.degree(n) > 0]
nx.draw_networkx_nodes(sub_pruned, pos, ax=ax2, nodelist=active_nodes, node_color=COLOR_NODE, node_size=100, alpha=0.8, edgecolors='white', linewidths=1.5)

# 画出被剪枝保留的精简连出线
edges_pruned = sub_pruned.edges(data=True)
weights_pruned = [min(2.0, 0.5 + d['weight'] * 4.0) for u, v, d in edges_pruned]
nx.draw_networkx_edges(sub_pruned, pos, ax=ax2, alpha=0.65, 
                       edge_color=COLOR_EDGE_PRUNED, width=1.5, 
                       arrows=True, arrowsize=10, connectionstyle='arc3,rad=0.15')
ax2.set_title(f"(b) Pruned Asymmetric Ego-Network", fontsize=18, fontweight='bold', pad=15)
ax2.axis('off')

# ---------- (c) 共现频次分布 ----------
ax3 = axes[1, 0]
from collections import Counter
freq_counts = Counter(HH_cooc.data)
valid_freqs = {f: c for f, c in freq_counts.items() if f > 0}
x_freq = np.array(list(valid_freqs.keys()))
y_cnt = np.array(list(valid_freqs.values()))

ax3.scatter(x_freq, y_cnt, color=COLOR_NODE, alpha=0.75, edgecolor='black', linewidth=0.6, s=50)
ax3.set_xscale('log')
ax3.set_yscale('log')
ax3.set_title("(c) Co-occurrence Frequency Distribution", fontsize=18, fontweight='bold', pad=15)
ax3.set_xlabel("Co-occurrence Frequency between Herbs", fontsize=15)
ax3.set_ylabel("Count of Herb Pairs", fontsize=15)
ax3.grid(True, which="both", ls="--", alpha=0.3)
ax3.tick_params(axis='both', labelsize=12)

# ---------- (d) 剪枝后出度分布 ----------
ax4 = axes[1, 1]
out_deg_counts = Counter([d for d in pruned_out_degrees if d > 0])
x_out = np.arange(1, MAX_NEIGHBORS + 1) 
y_out = [out_deg_counts.get(i, 0) for i in x_out]

ax4.bar(x_out, y_out, color=COLOR_NODE, edgecolor="black", alpha=0.8, width=0.7)
ax4.set_title(f"(d) Subgraph Out-Degree Distribution", fontsize=18, fontweight='bold', pad=15)
ax4.set_xlabel("Node Out-Degree", fontsize=15)
ax4.set_ylabel("Count of Herbs", fontsize=15)

ax4.set_xlim(0.5, MAX_NEIGHBORS + 0.5)
ax4.set_xticks(np.arange(1, MAX_NEIGHBORS + 1))
ax4.tick_params(axis='both', labelsize=12)
ax4.grid(True, axis='y', ls="--", alpha=0.3)

ax4.axvline(x=MAX_NEIGHBORS, color=COLOR_HUB, linestyle='--', linewidth=2.5, alpha=0.8)
ax_max_y = max(y_out) if y_out else 10
ax4.text(MAX_NEIGHBORS - 0.4, ax_max_y*0.9, f'$K_{{max}}={MAX_NEIGHBORS}$', color=COLOR_HUB, fontsize=16, fontweight='bold', ha='right')

# =========================================================
# 添加跨全局的高级精美图例 (Legend)
# =========================================================
marker_hub = mlines.Line2D([], [], color='w', marker='o', markerfacecolor=COLOR_HUB, markeredgecolor='black', markersize=14, label='Target Hub Node')
marker_node = mlines.Line2D([], [], color='w', marker='o', markerfacecolor=COLOR_NODE, markeredgecolor='white', markersize=12, label='Neighbor Herbs')
line_raw = mlines.Line2D([], [], color=COLOR_EDGE_RAW, linewidth=3, alpha=0.5, label='Raw Co-occurrence')
line_pruned = mlines.Line2D([], [], color=COLOR_EDGE_PRUNED, linewidth=2, marker='>', markersize=8, label=f'Top-{MAX_NEIGHBORS} Directed Edge')

fig.legend(handles=[marker_hub, marker_node, line_raw, line_pruned], 
           loc='upper center', ncol=4, bbox_to_anchor=(0.5, 0.98), 
           fontsize=14, frameon=True, shadow=False, edgecolor='gray')

plt.tight_layout(rect=[0, 0, 1, 0.94], pad=2.0) 
save_path_pdf = os.path.join(CURRENT_DIR, "anti_hub_Ginseng_Final_Publication.pdf")
save_path_png = os.path.join(CURRENT_DIR, "anti_hub_Ginseng_Final_Publication.png")
plt.savefig(save_path_pdf, dpi=300, bbox_inches='tight')
plt.savefig(save_path_png, dpi=300, bbox_inches='tight')
print("✅ Fully Reverted and Graph B Layout fixed successfully!")
