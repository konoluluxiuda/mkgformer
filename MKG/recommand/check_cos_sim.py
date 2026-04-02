import torch
import torch.nn.functional as F
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

with torch.no_grad():
    herb_idx = torch.tensor(data_manager.herb_indices, dtype=torch.long)
    st = F.normalize(model.embedding(herb_idx), dim=-1)
    at = F.normalize(model.attr_align(final_attr_matrix[herb_idx].float()), dim=-1)
    ch = F.normalize(model.chem_align(chem_matrix[herb_idx].float()), dim=-1)

    print("--- TRAINED MODEL (Cosine Similarity) ---")
    print(f"Sim ST-AT: {(st*at).sum(dim=-1).mean().item():.4f}")
    print(f"Sim ST-CH: {(st*ch).sum(dim=-1).mean().item():.4f}")
    print(f"Sim AT-CH: {(at*ch).sum(dim=-1).mean().item():.4f}")

model_untrained = HMC_GNN_SSL(
    num_nodes=data_manager.num_nodes,
    num_relations=data_manager.num_relations,
    pretrained_features=None,
    attr_matrix=final_attr_matrix,
    chem_matrix=chem_matrix,
    disease_matrix=None,
    fusion_mode='add'
)
model_untrained.eval()
with torch.no_grad():
    st_u = F.normalize(model_untrained.embedding(herb_idx), dim=-1)
    at_u = F.normalize(model_untrained.attr_align(final_attr_matrix[herb_idx].float()), dim=-1)
    ch_u = F.normalize(model_untrained.chem_align(chem_matrix[herb_idx].float()), dim=-1)

    print("\n--- UNTRAINED MODEL (Cosine Similarity) ---")
    print(f"Sim ST-AT: {(st_u*at_u).sum(dim=-1).mean().item():.4f}")
    print(f"Sim ST-CH: {(st_u*ch_u).sum(dim=-1).mean().item():.4f}")
    print(f"Sim AT-CH: {(at_u*ch_u).sum(dim=-1).mean().item():.4f}")

