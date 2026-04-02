import torch
import numpy as np
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
    st = model.embedding(herb_idx)
    at = model.attr_align(final_attr_matrix[herb_idx].float())
    ch = model.chem_align(chem_matrix[herb_idx].float())

    print("--- TRAINED MODEL ---")
    print(f"Norm ST: {st.norm(dim=1).mean().item():.4f}, AT: {at.norm(dim=1).mean().item():.4f}, CH: {ch.norm(dim=1).mean().item():.4f}")
    
    dist_sa = torch.pairwise_distance(st, at).mean().item()
    dist_sc = torch.pairwise_distance(st, ch).mean().item()
    dist_ac = torch.pairwise_distance(at, ch).mean().item()
    print(f"Dist ST-AT: {dist_sa:.4f}, ST-CH: {dist_sc:.4f}, AT-CH: {dist_ac:.4f}")

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
    st_u = model_untrained.embedding(herb_idx)
    at_u = model_untrained.attr_align(final_attr_matrix[herb_idx].float())
    ch_u = model_untrained.chem_align(chem_matrix[herb_idx].float())

    print("\n--- UNTRAINED MODEL ---")
    print(f"Norm ST: {st_u.norm(dim=1).mean().item():.4f}, AT: {at_u.norm(dim=1).mean().item():.4f}, CH: {ch_u.norm(dim=1).mean().item():.4f}")
    
    dist_sa_u = torch.pairwise_distance(st_u, at_u).mean().item()
    dist_sc_u = torch.pairwise_distance(st_u, ch_u).mean().item()
    dist_ac_u = torch.pairwise_distance(at_u, ch_u).mean().item()
    print(f"Dist ST-AT: {dist_sa_u:.4f}, ST-CH: {dist_sc_u:.4f}, AT-CH: {dist_ac_u:.4f}")

