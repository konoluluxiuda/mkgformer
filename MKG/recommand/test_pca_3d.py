import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from config import Config
from model import HMC_GNN_SSL
from dataset import GraphDataManager
import os
import torch.nn.functional as F

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

# Trained
model = HMC_GNN_SSL(num_nodes=data_manager.num_nodes, num_relations=data_manager.num_relations, pretrained_features=None, attr_matrix=final_attr_matrix, chem_matrix=chem_matrix, disease_matrix=None, fusion_mode='gated')
model.load_state_dict(torch.load('/home/zry/workspace/mkgformer/MKG/recommand/checkpoints/best_model.pt', map_location='cpu'), strict=False)
model.eval()

# Untrained
model_u = HMC_GNN_SSL(num_nodes=data_manager.num_nodes, num_relations=data_manager.num_relations, pretrained_features=None, attr_matrix=final_attr_matrix, chem_matrix=chem_matrix, disease_matrix=None, fusion_mode='add')
model_u.eval()

herb_idx = torch.tensor(data_manager.herb_indices, dtype=torch.long)
with torch.no_grad():
    st = F.normalize(model.embedding(herb_idx), dim=-1).numpy()
    at = F.normalize(model.attr_align(final_attr_matrix[herb_idx].float()), dim=-1).numpy()
    ch = F.normalize(model.chem_align(chem_matrix[herb_idx].float()), dim=-1).numpy()

    st_u = F.normalize(model_u.embedding(herb_idx), dim=-1).numpy()
    at_u = F.normalize(model_u.attr_align(final_attr_matrix[herb_idx].float()), dim=-1).numpy()
    ch_u = F.normalize(model_u.chem_align(chem_matrix[herb_idx].float()), dim=-1).numpy()

np.random.seed(42)
sample_mask = np.random.choice(len(herb_idx), 250, replace=False)

st, at, ch = st[sample_mask], at[sample_mask], ch[sample_mask]
st_u, at_u, ch_u = st_u[sample_mask], at_u[sample_mask], ch_u[sample_mask]

pca = PCA(n_components=3)
X_u = np.vstack((st_u, at_u, ch_u))
res_u = pca.fit_transform(X_u)

pca2 = PCA(n_components=3)
X = np.vstack((st, at, ch))
res = pca2.fit_transform(X)

fig = plt.figure(figsize=(14, 6))
ax1 = fig.add_subplot(121, projection='3d')
n = 250
ax1.scatter(res_u[:n,0], res_u[:n,1], res_u[:n,2], c='red', label='ST', s=10)
ax1.scatter(res_u[n:2*n,0], res_u[n:2*n,1], res_u[n:2*n,2], c='green', label='AT', s=10)
ax1.scatter(res_u[2*n:,0], res_u[2*n:,1], res_u[2*n:,2], c='blue', label='CH', s=10)
ax1.set_title('Untrained Normalized 3D PCA')

ax2 = fig.add_subplot(122, projection='3d')
ax2.scatter(res[:n,0], res[:n,1], res[:n,2], c='red', label='ST', s=10)
ax2.scatter(res[n:2*n,0], res[n:2*n,1], res[n:2*n,2], c='green', label='AT', s=10)
ax2.scatter(res[2*n:,0], res[2*n:,1], res[2*n:,2], c='blue', label='CH', s=10)
ax2.set_title('Trained Normalized 3D PCA')

plt.savefig('/home/zry/workspace/mkgformer/fig/test_pca_3d.png')
import sys
sys.exit(0)
