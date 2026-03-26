import os
import random
import pandas as pd
import numpy as np
from scipy import sparse
from collections import defaultdict
import matplotlib.pyplot as plt
import networkx as nx
import seaborn as sns

# =========================================================
# 1. 样式配置 (移除中文依赖，适应英文论文格式)
# =========================================================
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman', 'DejaVu Serif']
plt.rcParams['axes.unicode_minus'] = False 

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_ROOT = os.path.join(CURRENT_DIR, '..', 'dataset', 'NEWHERB', 'kge_data')
files = ['train.tsv', 'dev.tsv', 'test.tsv']

herb_disease_pairs = []
print(f"Loading KGE data from: {os.path.abspath(DATA_ROOT)}")

for f in files:
    path = os.path.join(DATA_ROOT, f)
    if os.path.exists(path):
        df = pd.read_csv(path, sep='\t', header=None, names=['h', 'r', 't'])
        hd = df[df['r'] == 'treats_disease']
        for _, row in hd.iterrows():
            herb_disease_pairs.append((row['h'], row['t']))

herbs = list(set([p[0] for p in herb_disease_pairs]))
diseases = list(set([p[1] for p in herb_disease_pairs]))

if len(herbs) == 0:
    print("❌ Error: No data loaded. Please check the dataset path.")
    exit(1)

h_idx = {h: i for i, h in enumerate(herbs)}
d_idx = {d: i for i, d in enumerate(diseases)}
print(f"Extracted {len(herbs)} Herbs and {len(diseases)} Diseases.")

# =========================================================
# 2. 构建 Herb-Disease 矩阵 & Herb-Herb 共现矩阵
# =========================================================
rows = [h_idx[p[0]] for p in herb_disease_pairs]
cols = [d_idx[p[1]] for p in herb_disease_pairs]
vals = [1] * len(rows)

H_D_mat = sparse.csr_matrix((vals, (rows, cols)), shape=(len(herbs), len(diseases)))

print("Computing Herb-Herb Co-occurrence matrix...")
HH_cooc = H_D_mat @ H_D_mat.T
HH_cooc.setdiag(0)
HH_cooc.eliminate_zeros()

# =========================================================
# 3. 模拟长尾去噪 / Top-K 截断
# =========================================================
MAX_NEIGHBORS = 15
MIN_FREQ = 1

raw_degrees = []
pruned_degrees = []
G_raw = nx.Graph()
G_pruned = nx.Graph()

print("Calculating degrees and building subgraphs...")
for i in range(len(herbs)):
    row = HH_cooc.getrow(i)
    valid_neighbors = [(j, val) for j, val in zip(row.indices, row.data) if val >= MIN_FREQ]
    raw_degrees.append(len(valid_neighbors))
    
    for j, val in valid_neighbors:
        if i < j: G_raw.add_edge(herbs[i], herbs[j], weight=val)
    
    valid_neighbors.sort(key=lambda x: x[1], reverse=True)
    pruned_neighbors = valid_neighbors[:MAX_NEIGHBORS]
    pruned_degrees.append(len(pruned_neighbors))
    
    for j, val in pruned_neighbors:
        if i < j: G_pruned.add_edge(herbs[i], herbs[j], weight=val)

print(f"Max Degree (Raw): {max(raw_degrees)}")
print(f"Max Degree (Pruned): {max(pruned_degrees)}")

# =========================================================
# 4. 指定 Hub 节点并画图
# =========================================================
# 寻找名为 "人参" 或 "HN1418" 的节点
TARGET_NAMES = ["人参", "HN1418", "Ginseng"]
hub_name = None

for name in TARGET_NAMES:
    if name in herbs:
        hub_name = name
        break

if hub_name is None:
    print(f"⚠️ Warning: Target hub not found. Falling back to the maximum degree node.")
    hub_idx = np.argmax(raw_degrees)
    hub_name = herbs[hub_idx]
else:
    hub_idx = h_idx[hub_name]

print(f"Selected Hub Node for Visualization: {hub_name} (Raw Degree: {raw_degrees[hub_idx]})")

fig, axes = plt.subplots(2, 2, figsize=(14, 12))

# 提取局部子图
raw_subgraph = nx.ego_graph(G_raw, hub_name, radius=1)
if len(raw_subgraph.nodes) > 100:
    neighbors = list(raw_subgraph.nodes)
    if hub_name in neighbors: neighbors.remove(hub_name)
    sampled = random.sample(neighbors, 100) + [hub_name]
    raw_subgraph = G_raw.subgraph(sampled)

pruned_subgraph = nx.ego_graph(G_pruned, hub_name, radius=1)

# -- (a) 原始子图 --
ax1 = axes[0, 0]
pos_raw = nx.spring_layout(raw_subgraph, seed=42, k=0.15)
nx.draw_networkx_nodes(raw_subgraph, pos_raw, ax=ax1, nodelist=[hub_name], node_color='red', node_size=300)
nx.draw_networkx_nodes(raw_subgraph, pos_raw, ax=ax1, nodelist=[n for n in raw_subgraph.nodes if n != hub_name], node_color='#999999', node_size=30, alpha=0.6)
nx.draw_networkx_edges(raw_subgraph, pos_raw, ax=ax1, alpha=0.1, edge_color='gray')
# 将显示名字强制转为英文以防止图表乱码
d_name = "Ginseng (HN1418)" if hub_name in ["人参", "HN1418"] else hub_name
ax1.set_title(f"(a) Raw Ego-Network Topology of '{d_name}'\n(Over-dense Hairball effect)", fontsize=16)
ax1.axis('off')

# -- (b) 剪枝后子图 --
ax2 = axes[0, 1]
pos_pruned = nx.spring_layout(pruned_subgraph, seed=42)
nx.draw_networkx_nodes(pruned_subgraph, pos_pruned, ax=ax2, nodelist=[hub_name], node_color='red', node_size=300)
nx.draw_networkx_nodes(pruned_subgraph, pos_pruned, ax=ax2, nodelist=[n for n in pruned_subgraph.nodes if n != hub_name], node_color='#4C72B0', node_size=80, alpha=0.9)
nx.draw_networkx_edges(pruned_subgraph, pos_pruned, ax=ax2, alpha=0.6, edge_color='#555555')
ax2.set_title(f"(b) Pruned Ego-Network (Top-K={MAX_NEIGHBORS})\n(Sparse and Confidence-driven)", fontsize=16)
ax2.axis('off')

# -- (c) 原始长尾度分布 --
ax3 = axes[1, 0]
sns.histplot(raw_degrees, bins=40, color="#C44E52", kde=True, ax=ax3)
ax3.set_title("(c) Degree Distribution (Raw Graph)", fontsize=16)
ax3.set_xlabel("Node Degree (Count of co-occurring herbs)")
ax3.set_ylabel("Count of Herbs")
ax3.set_yscale('log')
ax3.text(0.45, 0.8, 'Strong Popularity Bias', transform=ax3.transAxes, color='red', fontsize=14, fontweight='bold')

# -- (d) 剪枝后度分布 --
ax4 = axes[1, 1]
sns.histplot(pruned_degrees, bins=15, color="#55A868", discrete=True, ax=ax4)
ax4.set_title(f"(d) Degree Distribution (Pruned to Top-{MAX_NEIGHBORS})", fontsize=16)
ax4.set_xlabel("Node Degree Constraint")
ax4.set_ylabel("Count of Herbs")

plt.tight_layout()
save_path_pdf = os.path.join(CURRENT_DIR, "anti_hub_Ginseng.pdf")
save_path_png = os.path.join(CURRENT_DIR, "anti_hub_Ginseng.png")
plt.savefig(save_path_pdf, dpi=300, bbox_inches='tight')
plt.savefig(save_path_png, dpi=300, bbox_inches='tight')
print(f"✅ Visualization saved at:\n{save_path_pdf}\n{save_path_png}")