import os
import random
import pandas as pd
import numpy as np
from scipy import sparse
import matplotlib.pyplot as plt
import networkx as nx
import seaborn as sns

# =========================================================
# 1. 样式配置
# =========================================================
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman', 'DejaVu Serif']
plt.rcParams['axes.unicode_minus'] = False 

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_ROOT = os.path.join(CURRENT_DIR, '..', 'dataset', 'NEWHERB', 'kge_data')

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
# 2. 构建 Herb-Disease 矩阵 & Herb-Herb 共现矩阵
# =========================================================
rows = [h_idx[p[0]] for p in herb_disease_pairs]
cols = [d_idx[p[1]] for p in herb_disease_pairs]
vals = [1] * len(rows)

H_D_mat = sparse.csr_matrix((vals, (rows, cols)), shape=(len(herbs), len(diseases)))
HH_cooc = H_D_mat @ H_D_mat.T
HH_cooc.setdiag(0)
HH_cooc.eliminate_zeros()

# =========================================================
# 3. 模拟长尾去噪 / Top-K 截断
# =========================================================
MAX_NEIGHBORS = 15
MIN_FREQ = 2

raw_degrees = []
pruned_degrees = []
G_raw = nx.Graph()
G_pruned = nx.Graph()

for i in range(len(herbs)):
    row = HH_cooc.getrow(i)
    # 取全部出现的边为基础绘制长尾
    all_neighbors = [(j, val) for j, val in zip(row.indices, row.data)]
    raw_degrees.append(len(all_neighbors))  # 保持原始图长尾
    
    valid_neighbors = [n for n in all_neighbors if n[1] >= MIN_FREQ]
    for j, val in valid_neighbors:
        if i < j: G_raw.add_edge(herbs[i], herbs[j], weight=val)
    
    valid_neighbors.sort(key=lambda x: x[1], reverse=True)
    pruned_neighbors = valid_neighbors[:MAX_NEIGHBORS]
    pruned_degrees.append(len(pruned_neighbors))
    
    for j, val in pruned_neighbors:
        if i < j: G_pruned.add_edge(herbs[i], herbs[j], weight=val)

# =========================================================
# 4. 指定 Hub 节点并画图
# =========================================================
TARGET_NAMES = ["人参", "HN1418", "Ginseng"]
hub_name = next((name for name in TARGET_NAMES if name in herbs), herbs[np.argmax(raw_degrees)])

fig, axes = plt.subplots(2, 2, figsize=(16, 14))

# 提取局部子图
raw_subgraph = nx.ego_graph(G_raw, hub_name, radius=1)
if len(raw_subgraph.nodes) > 100:
    neighbors = list(raw_subgraph.nodes)
    if hub_name in neighbors: neighbors.remove(hub_name)
    sampled = random.sample(neighbors, 100) + [hub_name]
    raw_subgraph = G_raw.subgraph(sampled)

pruned_subgraph = nx.ego_graph(G_pruned, hub_name, radius=1)
d_name = "Ginseng (HN1418)" if hub_name in ["人参", "HN1418"] else hub_name

# ---------- (a) 原始子图 (解决黑块问题) ----------
ax1 = axes[0, 0]
pos_raw = nx.spring_layout(raw_subgraph, seed=42, k=0.2)
nx.draw_networkx_nodes(raw_subgraph, pos_raw, ax=ax1, nodelist=[hub_name], node_color='#D62728', node_size=300, edgecolors='black')
nx.draw_networkx_nodes(raw_subgraph, pos_raw, ax=ax1, nodelist=[n for n in raw_subgraph.nodes if n != hub_name], node_color='#CCCCCC', node_size=20, alpha=0.6)

# 强制截断并缩小线宽！
edges_raw = raw_subgraph.edges(data=True)
weights_raw = [min(0.8, d['weight'] * 0.05) for u, v, d in edges_raw] # 限制最大线宽不超过0.8
nx.draw_networkx_edges(raw_subgraph, pos_raw, ax=ax1, alpha=0.1, edge_color='gray', width=weights_raw)

ax1.set_title(f"(a) Raw Ego-Network Topology of '{d_name}'\n(Over-dense Hairball effect)", fontsize=18, fontweight='bold', pad=15)
ax1.axis('off')

# ---------- (b) 剪枝后子图 (解决黑块问题) ----------
ax2 = axes[0, 1]
pos_pruned = nx.spring_layout(pruned_subgraph, seed=42, k=0.35)
nx.draw_networkx_nodes(pruned_subgraph, pos_pruned, ax=ax2, nodelist=[hub_name], node_color='#D62728', node_size=300, edgecolors='black')
nx.draw_networkx_nodes(pruned_subgraph, pos_pruned, ax=ax2, nodelist=[n for n in pruned_subgraph.nodes if n != hub_name], node_color='#4C72B0', node_size=70, alpha=0.9, edgecolors='white')

edges_pruned = pruned_subgraph.edges(data=True)
# 这里放大一点展示信心度，但也限制上限
weights_pruned = [min(2.5, d['weight'] * 0.1) for u, v, d in edges_pruned]
nx.draw_networkx_edges(pruned_subgraph, pos_pruned, ax=ax2, alpha=0.3, edge_color='#444444', width=weights_pruned)

ax2.set_title(f"(b) Pruned Ego-Network (Top-K={MAX_NEIGHBORS})\n(Sparse and Confidence-driven)", fontsize=18, fontweight='bold', pad=15)
ax2.axis('off')

# ---------- (c) 原始长尾度分布 (修正) ----------
ax3 = axes[1, 0]
# 将所有共现频次拿出来展示经典的 Power-law
all_freqs = HH_cooc.data
sns.histplot(all_freqs, bins=50, color="#C44E52", log_scale=(True, True), ax=ax3)

ax3.set_title("(c) Co-occurrence Frequency Distribution", fontsize=18, fontweight='bold', pad=15)
ax3.set_xlabel("Co-occurrence Frequency between Herbs (Log Scale)", fontsize=14)
ax3.set_ylabel("Count of Edge Pairs (Log Scale)", fontsize=14)

ax3.text(0.5, 0.8, 'Strong Popularity Bias\n(Classic Power-law)', transform=ax3.transAxes, 
         color='#D62728', fontsize=16, fontweight='bold', ha='center')

# ---------- (d) 剪枝后度分布 ----------
ax4 = axes[1, 1]
sns.histplot(pruned_degrees, bins=range(0, 22), color="#55A868", discrete=True, ax=ax4, edgecolor="black")
ax4.set_title(f"(d) Subgraph Node Degree Distribution (Max={MAX_NEIGHBORS})", fontsize=18, fontweight='bold', pad=15)
ax4.set_xlabel("Node Degree Capacity Constraint", fontsize=14)
ax4.set_ylabel("Count of Herbs", fontsize=14)
ax4.set_xlim(0, 20)
ax4.set_xticks(range(0, 21, 5))

ax4.text(0.2, 0.7, r'Information Bottleneck ($MAX\_K \leq 15$)', transform=ax4.transAxes, 
         color='#1B5E20', fontsize=16, fontweight='bold')

plt.tight_layout(pad=3.0)
save_path_pdf = os.path.join(CURRENT_DIR, "anti_hub_Ginseng_Optimized_V2.pdf")
save_path_png = os.path.join(CURRENT_DIR, "anti_hub_Ginseng_Optimized_V2.png")
plt.savefig(save_path_pdf, dpi=300, bbox_inches='tight')
plt.savefig(save_path_png, dpi=300, bbox_inches='tight')
print("✅ Fully Optimized Visualization V2 saved!")