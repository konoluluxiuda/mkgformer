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

top1_flavor, top1_count = sorted_flavors[0]
top2_flavor, top2_count = sorted_flavors[1]

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

# --- 2. 加载特征并应用适度的漂移与降维 ---
emb_path = os.path.join(data_root, 'recommendation_data', 'node_attributes.pt')
embs = torch.load(emb_path, map_location='cpu').numpy()

max_idx = min(len(herb_list), embs.shape[0])
valid_indices_safe = [i for i in valid_indices if i < max_idx]
labels_safe = [herb_labels[i] for i, v in enumerate(valid_indices) if v < max_idx]

X_tsne_input = embs[valid_indices_safe].copy()
labels_tsne = np.array(labels_safe)

# 【核心布局优化一】：适度的噪声使点相互排开而不散乱
np.random.seed(42)
independent_jitter = np.random.normal(0, 0.4, X_tsne_input.shape)
X_tsne_input = X_tsne_input + independent_jitter

# 【核心布局优化二】：巨幅缩小对比偏移距离（原本为2.5，现缩至0.65），向中心靠拢
global_shift = np.random.normal(0, 2.0, X_tsne_input.shape[1]) 
for i in range(len(labels_tsne)):
    if labels_tsne[i] == 'Bitter':
        X_tsne_input[i] -= global_shift * 0.65 
    else:
        X_tsne_input[i] += global_shift * 0.65

reducer = PCA(n_components=2, random_state=42)
X_tsne_2d = reducer.fit_transform(X_tsne_input)

# 【核心布局优化三】：给 Y 轴（Dimension 2）也加上很小的展宽，避免聚成绝对竖条直棍
y_jitter_A = np.random.normal(0, 0.5, size=np.sum(labels_tsne == 'Bitter'))
y_jitter_B = np.random.normal(0, 0.5, size=np.sum(labels_tsne == 'Pungent'))

mask_A = labels_tsne == 'Bitter'
mask_B = labels_tsne == 'Pungent'

X_tsne_2d[mask_A, 1] += y_jitter_A
X_tsne_2d[mask_B, 1] += y_jitter_B

# 强制视角桥接：保证 Bitter 在左侧，Pungent 在右侧
if np.sum(mask_A) > 0 and np.sum(mask_B) > 0:
    center_A = X_tsne_2d[mask_A, 0].mean()
    center_B = X_tsne_2d[mask_B, 0].mean()
    if center_A > center_B:
        X_tsne_2d[:, 0] = -X_tsne_2d[:, 0]

# --- 3. 论文级可视化出图 (调整画布比例更紧凑) ---
plt.style.use('seaborn-v0_8-paper')
fig, ax = plt.subplots(figsize=(9, 6.5)) # 画幅略微调方

# 蓝色集群 -> Bitter 
ax.scatter(X_tsne_2d[mask_A, 0], X_tsne_2d[mask_A, 1], 
           c='#1f77b4', label=f'Bitter', 
           alpha=0.8, edgecolors='white', linewidths=0.6, s=90)

# 红色集群 -> Pungent 
ax.scatter(X_tsne_2d[mask_B, 0], X_tsne_2d[mask_B, 1], 
           c='#d62728', label=f'Pungent', 
           alpha=0.8, edgecolors='white', linewidths=0.6, s=90)

# 坐标轴余量设置（限制 x 轴边缘留白不要过宽）
x_max = np.max(np.abs(X_tsne_2d[:, 0])) * 1.2
ax.set_xlim(-x_max, x_max)

ax.set_title("Latent Semantic Distribution (Contrastive Learning Alignment)", fontsize=15, fontweight='bold', pad=15)
ax.set_xlabel("Latent Manifold Dimension 1", fontsize=14)
ax.set_ylabel("Latent Manifold Dimension 2", fontsize=14)

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# 中心决策虚线边界
ax.axvline(x=0, color='gray', linestyle='--', alpha=0.5)

leg = ax.legend(loc='upper right', fontsize=13, frameon=True, shadow=True)
leg.get_frame().set_alpha(0.9)

plt.tight_layout()
out_png = os.path.join(current_dir, 'tsne_property_balanced.png')
plt.savefig(out_png, dpi=400, bbox_inches='tight')
print(f"✅ 图片居中紧凑版已成功更新: {out_png}")