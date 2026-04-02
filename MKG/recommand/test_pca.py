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

model = HMC_GNN_SSL(num_nodes=data_manager.num_nodes, num_relations=data_manager.num_relations, pretrained_features=None, attr_matrix=final_attr_matrix, chem_matrix=chem_matrix, disease_matrix=None, fusion_mode='gated')
model.load_state_dict(torch.load('/home/zry/workspace/mkgformer/MKG/recommand/checkpoints/best_model.pt', map_location='cpu'), strict=False)
model.eval()

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

X_u = np.vstack((st_u, at_u, ch_u))
pca_u = PCA(n_components=2).fit_transform(X_u)

X = np.vstack((st, at, ch))
pca = PCA(n_components=2).fit_transform(X)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
n = len(sample_mask)
print(f"Untrained PCA variance ratio: {PCA(n_components=2).fit(X_u).explained_variance_ratio_}")
print(f"Trained PCA variance ratio: {PCA(n_components=2).fit(X).explained_variance_ratio_}")

# Let's compute center of mass
print(f"Untrained Centers: ST= {pca_u[:n].mean(0)}, AT= {pca_u[n:2*n].mean(0)}, CH= {pca_u[2*n:].mean(0)}")
print(f"Trained Centers: ST= {pca[:n].mean(0)}, AT= {pca[n:2*n].mean(0)}, CH= {pca[2*n:].mean(0)}")

