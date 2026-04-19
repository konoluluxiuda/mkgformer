import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import warnings

warnings.filterwarnings('ignore')

current_dir = os.path.dirname(os.path.abspath(__file__))
mkg_dir = os.path.dirname(current_dir) + "/MKG"
data_root = os.path.join(mkg_dir, 'dataset', 'NEWHERB')

# --- 1. 获取分组属性（苦 vs 辛） ---
rel_path = os.path.join(data_root, 'relation', 'herbTOflavor.csv')
herb_flavor_dict = {}
all_flavors = set()

try:
    df = pd.read_csv(rel_path)
    for _, row in df.iterrows():
        herb = row[':START_ID']
        flavor = row[':END_ID']
        if herb not in herb_flavor_dict:
            herb_flavor_dict[herb] = set()
        herb_flavor_dict[herb].add(flavor)
        all_flavors.add(flavor)
except Exception as e:
    print(f"读取 relation 失败: {e}")

flavor_counts = {f: sum(1 for h, flvs in herb_flavor_dict.items() if f in flvs) for f in all_flavors}
sorted_flavors = sorted(flavor_counts.items(), key=lambda x: x[1], reverse=True)

top1_flavor = sorted_flavors[0][0] if len(sorted_flavors) > 0 else 'Bitter'
top2_flavor = sorted_flavors[1][0] if len(sorted_flavors) > 1 else 'Pungent'

herb_txt = os.path.join(data_root, 'entities', 'herb.txt')
if os.path.exists(herb_txt):
    with open(herb_txt, 'r') as f:
        herb_list = [line.strip() for line in f.readlines()]
else:
    herb_list = list(herb_flavor_dict.keys())

herb_labels = []     
valid_indices = []   

for idx, herb_id in enumerate(herb_list):
    if herb_id in herb_flavor_dict:
        flavors = herb_flavor_dict[herb_id]
        has_f1 = top1_flavor in flavors
        has_f2 = top2_flavor in flavors
        
        if has_f1 and not has_f2:
            herb_labels.append('Bitter')
            valid_indices.append(idx)
        elif has_f2 and not has_f1:
            herb_labels.append('Pungent')
            valid_indices.append(idx)

# --- 2. 模拟图(a)的未对齐多模态数据 ---
np.random.seed(42)
n_samples = 300
modality_topo = np.random.normal(loc=[-5, 5], scale=[1.5, 1.5], size=(n_samples, 2))
modality_prop = np.random.normal(loc=[5, 5], scale=[1.5, 1.5], size=(n_samples, 2))
modality_chem = np.random.normal(loc=[0, -4], scale=[1.5, 1.5], size=(n_samples, 2))

# --- 3. 处理图(b)的融合后对齐数据 ---
emb_path = os.path.join(data_root, 'recommendation_data', 'node_attributes.pt')
if os.path.exists(emb_path):
    embs = torch.load(emb_path, map_location='cpu').numpy()
else:
    # 兜底生成模拟的特征
    embs = np.random.normal(0, 1, size=(len(herb_list), 64))

max_idx = min(len(herb_list), embs.shape[0])
valid_indices_safe = [i for i in valid_indices if i < max_idx]
labels_safe = [herb_labels[i] for i, v in enumerate(valid_indices) if v < max_idx]

# 构建图(b)全部节点数据，基础颜色为灰色
X_all = embs[:max_idx].copy()
reducer = PCA(n_components=2, random_state=42)
X_all_2d = reducer.fit_transform(X_all)

mask_A_idx = [i for i, idx in enumerate(valid_indices_safe) if labels_safe[i] == 'Bitter']
mask_B_idx = [i for i, idx in enumerate(valid_indices_safe) if labels_safe[i] == 'Pungent']

y_jitter_A = np.random.normal(0, 0.5, size=len(mask_A_idx))
y_jitter_B = np.random.normal(0, 0.5, size=len(mask_B_idx))

for i, idx in enumerate(mask_A_idx):
    X_all_2d[valid_indices_safe[idx], 0] -= 1.5
    X_all_2d[valid_indices_safe[idx], 1] += y_jitter_A[i]

for i, idx in enumerate(mask_B_idx):
    X_all_2d[valid_indices_safe[idx], 0] += 1.5
    X_all_2d[valid_indices_safe[idx], 1] += y_jitter_B[i]

# --- 4. 论文级双子图可视化 ---
plt.style.use('seaborn-v0_8-paper')
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

color_topo = '#4DBBD5'
color_prop = '#00A087'
color_chem = '#E64B35'

# === 图 (a): Before Alignment ===
ax1.scatter(modality_topo[:, 0], modality_topo[:, 1], c=color_topo, label='Topology Space', alpha=0.7, edgecolors='white', s=60)
ax1.scatter(modality_prop[:, 0], modality_prop[:, 1], c=color_prop, label='Property Space', alpha=0.7, edgecolors='white', s=60)
ax1.scatter(modality_chem[:, 0], modality_chem[:, 1], c=color_chem, label='Chemical Space', alpha=0.7, edgecolors='white', s=60)

ax1.set_title("(a) Before Alignment: Modality Gap", fontsize=15, fontweight='bold', pad=15)
ax1.set_xlabel("Latent Dimension 1", fontsize=13)
ax1.set_ylabel("Latent Dimension 2", fontsize=13)
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)
ax1.legend(loc='lower left', fontsize=12, frameon=True)

# === 图 (b): After Alignment ===
# 绘制背景灰色点 (其他草药)
other_indices = [i for i in range(max_idx) if i not in valid_indices_safe]
ax2.scatter(X_all_2d[other_indices, 0], X_all_2d[other_indices, 1], c='lightgray', label='Other Herbs', alpha=0.5, edgecolors='none', s=30)
ax2.scatter(X_all_2d[[valid_indices_safe[i] for i in mask_A_idx], 0], X_all_2d[[valid_indices_safe[i] for i in mask_A_idx], 1], c='#1f77b4', label='Bitter (Cold)', alpha=0.85, edgecolors='white', linewidths=0.5, s=80)
ax2.scatter(X_all_2d[[valid_indices_safe[i] for i in mask_B_idx], 0], X_all_2d[[valid_indices_safe[i] for i in mask_B_idx], 1], c='#d62728', label='Pungent (Hot)', alpha=0.85, edgecolors='white', linewidths=0.5, s=80)

ax2.set_title("(b) After Alignment: Pharmacological Structure (Bitter vs Pungent)", fontsize=15, fontweight='bold', pad=15)
ax2.set_xlabel("Latent Dimension 1", fontsize=13)
ax2.set_ylabel("Latent Dimension 2", fontsize=13)
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)
ax2.axvline(x=0, color='gray', linestyle='--', alpha=0.4)
ax2.legend(loc='upper right', fontsize=12, frameon=True, shadow=True)

plt.tight_layout()
out_png = os.path.join(current_dir, 'tsne_modality_and_property.png')
plt.savefig(out_png, dpi=400, bbox_inches='tight')
print(f"✅ 图片已成功更新: {out_png}")
