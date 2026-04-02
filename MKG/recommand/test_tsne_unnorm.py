import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from config import Config
from model import HMC_GNN_SSL
from dataset import GraphDataManager
import os

Config.REC_DATA_DIR = os.path.join(Config.DATA_ROOT, 'paper_graph_data')
data_manager = GraphDataManager()
data_manager.load_data()  
    
chem_matrix = torch.load(os.path.join(Config.DATA_ROOT, 'recommendation_data', 'node_chem_dense.pt'))
fp_path = os.path.join(Config.DATA_ROOT, 'recommendation_data', 'node_chem_fingerprint.pt')
if os.path.exists(fp_path):
    chem_matrix = torch.cat([chem_matrix, torch.load(fp_path)], dim=1)

attr_tensors = []
base_attr = data_manager.load_attributes()
if base_attr is not None: attr_tensors.append(base_attr)
chem_attr = torch.load(os.path.join(Config.DATA_ROOT, 'recommendation_data', 'node_chem_multihot.pt'))
if chem_attr is not None: attr_tensors.append(chem_attr)
final_attr_matrix = torch.cat(attr_tensors, dim=1)

model = HMC_GNN_SSL(
    num_nodes=data_manager.num_nodes,
    num_relations=data_manager.num_relations,
    pretrained_features=None,
    attr_matrix=final_attr_matrix,
    chem_matrix=chem_matrix,
    disease_matrix=None,
    fusion_mode='gated'
)
model.load_state_dict(torch.load('/home/zry/workspace/mkgformer/MKG/recommand/checkpoints/best_model.pt', map_location='cpu'), strict=False)
model.eval()

model_u = HMC_GNN_SSL(num_nodes=data_manager.num_nodes, num_relations=data_manager.num_relations, pretrained_features=None, attr_matrix=final_attr_matrix, chem_matrix=chem_matrix, disease_matrix=None, fusion_mode='add')
model_u.eval()

herb_idx = torch.tensor(data_manager.herb_indices, dtype=torch.long)
with torch.no_grad():
    st = model.embedding(herb_idx).numpy()
    at = model.attr_align(final_attr_matrix[herb_idx].float()).numpy()
    ch = model.chem_align(chem_matrix[herb_idx].float()).numpy()

    st_u = model_u.embedding(herb_idx).numpy()
    at_u = model_u.attr_align(final_attr_matrix[herb_idx].float()).numpy()
    ch_u = model_u.chem_align(chem_matrix[herb_idx].float()).numpy()

np.random.seed(42)
sample_mask = np.random.choice(len(herb_idx), 250, replace=False)

st, at, ch = st[sample_mask], at[sample_mask], ch[sample_mask]
st_u, at_u, ch_u = st_u[sample_mask], at_u[sample_mask], ch_u[sample_mask]

X_u = np.vstack((st_u, at_u, ch_u))
tsne_u = TSNE(n_components=2, perplexity=30, random_state=42, init='pca').fit_transform(X_u)

X = np.vstack((st, at, ch))
tsne = TSNE(n_components=2, perplexity=30, random_state=42, init='pca').fit_transform(X)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
n = len(sample_mask)
ax1.scatter(tsne_u[:n,0], tsne_u[:n,1], label='ST', s=10)
ax1.scatter(tsne_u[n:2*n,0], tsne_u[n:2*n,1], label='AT', s=10)
ax1.scatter(tsne_u[2*n:,0], tsne_u[2*n:,1], label='CH', s=10)
ax1.set_title('Untrained Unnorm')

ax2.scatter(tsne[:n,0], tsne[:n,1], label='ST', s=10)
ax2.scatter(tsne[n:2*n,0], tsne[n:2*n,1], label='AT', s=10)
ax2.scatter(tsne[2*n:,0], tsne[2*n:,1], label='CH', s=10)
ax2.set_title('Trained Unnorm')
ax2.legend()
plt.savefig('/home/zry/workspace/mkgformer/fig/test_tsne.png')
import sys
sys.exit(0)
