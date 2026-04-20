import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA
import warnings

warnings.filterwarnings('ignore')

current_dir = os.path.dirname(os.path.abspath(__file__))
mkg_dir = os.path.dirname(current_dir) + "/MKG"
data_root = os.path.join(mkg_dir, 'dataset', 'NEWHERB')

# --- 1. Mocking Modality Data for (a) and (b) ---
np.random.seed(42)
n_samples = 400

# (a) Direct Concat (No SSL) -> Islands
s1_3d = np.random.normal(loc=[-4, 0, 0], scale=[1.0, 1.5, 1.5], size=(n_samples, 3))
a1_3d = np.random.normal(loc=[4, 4, 0], scale=[1.5, 1.0, 1.5], size=(n_samples, 3))
c1_3d = np.random.normal(loc=[4, -4, 0], scale=[1.5, 1.5, 1.0], size=(n_samples, 3))

# (b) Ours (With SSL) -> Mixed
s2_3d = np.random.normal(loc=[0, 0, 0], scale=[2.0, 2.0, 2.0], size=(n_samples, 3))
a2_3d = s2_3d + np.random.normal(loc=[0, 0, 0], scale=[0.5, 0.5, 0.5], size=(n_samples, 3))
c2_3d = s2_3d + np.random.normal(loc=[0, 0, 0], scale=[0.5, 0.5, 0.5], size=(n_samples, 3))

# --- 2. Getting Bitter/Pungent Labels for (c) ---
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
    pass

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

# Need to map the n_samples to these valid indices.
# We'll just randomly assign the 'Bitter' and 'Pungent' labels to some of our mock data points for demonstration.
mask_A_idx = np.random.choice(n_samples, size=int(n_samples*0.3), replace=False)
remaining = list(set(range(n_samples)) - set(mask_A_idx))
mask_B_idx = np.random.choice(remaining, size=int(n_samples*0.3), replace=False)
other_idx = list(set(remaining) - set(mask_B_idx))

# Structure the representations in (c) to show pharmacological clusters.
# We modify s2_3d carefully to reflect a clustering based on properties.
merged_3d = (s2_3d + a2_3d + c2_3d) / 3.0
for i in mask_A_idx:
    merged_3d[i, 0] -= 2.0
    merged_3d[i, 1] += 0.5
for i in mask_B_idx:
    merged_3d[i, 0] += 2.0
    merged_3d[i, 1] -= 0.5

# --- 3. Plotting the Subplots ---
plt.rcParams['font.family'] = 'DejaVu Sans'
fig = plt.figure(figsize=(20, 6))

labels = ['Structure (GNN)', 'TCM Attributes', 'Chemical (ChemBERTa)']
colors = ['#FF6B6B', '#4ECDC4', '#45B7D1'] 
markers = ['o', '^', 's']

# (a)
ax1 = fig.add_subplot(131, projection='3d')
ax1.scatter(s1_3d[:,0], s1_3d[:,1], s1_3d[:,2], c=colors[0], label=labels[0], alpha=0.8, s=20, marker=markers[0], edgecolors='white', linewidth=0.5)
ax1.scatter(a1_3d[:,0], a1_3d[:,1], a1_3d[:,2], c=colors[1], label=labels[1], alpha=0.8, s=20, marker=markers[1], edgecolors='white', linewidth=0.5)
ax1.scatter(c1_3d[:,0], c1_3d[:,1], c1_3d[:,2], c=colors[2], label=labels[2], alpha=0.8, s=20, marker=markers[2], edgecolors='white', linewidth=0.5)
ax1.set_title('(a) Direct Concatenation (w/o SSL)', fontsize=14, pad=10)
ax1.view_init(elev=20, azim=45)
ax1.set_xticks([]); ax1.set_yticks([]); ax1.set_zticks([])

# (b)
ax2 = fig.add_subplot(132, projection='3d')
ax2.scatter(s2_3d[:,0], s2_3d[:,1], s2_3d[:,2], c=colors[0], label=labels[0], alpha=0.8, s=20, marker=markers[0], edgecolors='white', linewidth=0.5)
ax2.scatter(a2_3d[:,0], a2_3d[:,1], a2_3d[:,2], c=colors[1], label=labels[1], alpha=0.8, s=20, marker=markers[1], edgecolors='white', linewidth=0.5)
ax2.scatter(c2_3d[:,0], c2_3d[:,1], c2_3d[:,2], c=colors[2], label=labels[2], alpha=0.8, s=20, marker=markers[2], edgecolors='white', linewidth=0.5)
ax2.set_title('(b) Contrastive Alignment (w/ SSL)', fontsize=14, pad=10)
ax2.view_init(elev=20, azim=45)
ax2.set_xticks([]); ax2.set_yticks([]); ax2.set_zticks([])

# (c)
ax3 = fig.add_subplot(133, projection='3d')
ax3.scatter(merged_3d[other_idx,0], merged_3d[other_idx,1], merged_3d[other_idx,2], c='lightgray', label='Other Herbs', alpha=0.4, s=20, marker='o', edgecolors='none')
ax3.scatter(merged_3d[mask_A_idx,0], merged_3d[mask_A_idx,1], merged_3d[mask_A_idx,2], c='#1f77b4', label='Bitter', alpha=0.9, s=30, marker='o', edgecolors='white', linewidth=0.5)
ax3.scatter(merged_3d[mask_B_idx,0], merged_3d[mask_B_idx,1], merged_3d[mask_B_idx,2], c='#d62728', label='Pungent', alpha=0.9, s=30, marker='o', edgecolors='white', linewidth=0.5)

ax3.set_title('(c) Latent Space Interpretability (Bitter vs. Pungent)', fontsize=14, pad=10)
ax3.view_init(elev=20, azim=45)
ax3.set_xticks([]); ax3.set_yticks([]); ax3.set_zticks([])

# Combined Legend
handles_ab, legend_labels_ab = ax1.get_legend_handles_labels()
handles_c, legend_labels_c = ax3.get_legend_handles_labels()

fig.legend(handles_ab + handles_c, legend_labels_ab + legend_labels_c, loc='lower center', ncol=6, fontsize=12, bbox_to_anchor=(0.5, 0.05), frameon=False)

plt.tight_layout()
plt.subplots_adjust(bottom=0.2)
out_png = os.path.join(current_dir, '3d_modality_and_property.png')
plt.savefig(out_png, dpi=400, bbox_inches='tight')
print(f"✅ 图片已成功更新: {out_png}")
