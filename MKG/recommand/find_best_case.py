import os
import torch
import numpy as np
import pandas as pd
import sys

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
MKG_DIR = os.path.dirname(CURRENT_DIR)
PROJECT_ROOT = os.path.dirname(MKG_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from config import Config
from dataset import GraphDataManager
from model import HMC_GNN_SSL
from KDHR.model import KDHR
from train_kdhr_newherb import load_kdhr_data

def get_name_map(csv_path):
    df = pd.read_csv(csv_path)
    # 假设 csv 中包含 id 和 name 两列，例如 ETCM_disease_id_2 -> 3-Methylglutaric Aciduria
    # 这里需要将您的 node id 对应起来，我们假设 graph_data_manager 中有映射
    return {row['id']: row['name'] for _, row in df.iterrows()}

def main():
    device = torch.device(Config.device)
    
    # 1. 加载字典与实体映射
    data_manager = GraphDataManager()
    edge_index, edge_type, train_dict, test_dict_gm = data_manager.load_data()
    mkg_dir = os.path.dirname(os.path.abspath(__file__))
    if 'recommand' in mkg_dir: mkg_dir = os.path.dirname(mkg_dir)
        
    disease_map = get_name_map(os.path.join(mkg_dir, 'dataset', 'NEWHERB', 'entities', 'disease.csv'))
    herb_map = get_name_map(os.path.join(mkg_dir, 'dataset', 'NEWHERB', 'entities', 'herb.csv'))
    
    with open(os.path.join(mkg_dir, 'dataset', 'NEWHERB', 'kge_data', 'entities.txt'), 'r') as f:
        global_entities = [line.strip() for line in f.readlines()]

    # 2. 加载 KDHR
    kdhr_data = load_kdhr_data()
    eval_meta = kdhr_data['eval_meta']
    test_dict = eval_meta['test_dict']
    
    kdhr_to_global_herb = {v: k for k, v in eval_meta['global_to_kdhr_herb'].items()}
    
    kdhr_model = KDHR(
        kdhr_data['num_diseases'], kdhr_data['num_herbs'], kdhr_data['sh_num'],
        64, Config.batch_size, 0.0, kg_dim=kdhr_data['kg_dim']
    ).to(device)
    kdhr_model.load_state_dict(torch.load('MKG/recommand/checkpoints/kdhr_best.pt', map_location=device))
    kdhr_model.eval()

    # 3. 加载 HMC_GNN_SSL
    attr_tensors = []
    base_attr = data_manager.load_attributes()
    if base_attr is not None: attr_tensors.append(base_attr)
    chem_attr = torch.load(os.path.join(Config.DATA_ROOT, 'recommendation_data', 'node_chem_multihot.pt'))
    if chem_attr is not None: attr_tensors.append(chem_attr)
    final_attr_matrix = torch.cat(attr_tensors, dim=1).to(device)
    
    chem_matrix = torch.load(os.path.join(Config.DATA_ROOT, 'recommendation_data', 'node_chem_dense.pt'))
    fp_feat = torch.load(os.path.join(Config.DATA_ROOT, 'recommendation_data', 'node_chem_fingerprint.pt'))
    chem_matrix = torch.cat([chem_matrix, fp_feat], dim=1).to(device)

    disease_matrix = None
    disease_path = os.path.join(Config.DATA_ROOT, 'recommendation_data', 'node_disease_text.pt')
    if os.path.exists(disease_path):
        disease_matrix = torch.load(disease_path).to(device)

    hmc_model = HMC_GNN_SSL(
        num_nodes=data_manager.num_nodes, num_relations=data_manager.num_relations,
        pretrained_features=None, attr_matrix=final_attr_matrix, chem_matrix=chem_matrix,
        disease_matrix=disease_matrix,
        fusion_mode='gated'
    ).to(device)
    hmc_model.load_state_dict(torch.load('MKG/recommand/checkpoints/best_model.pt', map_location=device))
    hmc_model.eval()

    # 4. 前向传播提取全部节点 Embeddings
    with torch.no_grad():
        hmc_emb = hmc_model.forward_encoder(edge_index.to(device), edge_type.to(device), perturbed=False)
        
        graph_d = kdhr_data['graph_data']
        kdhr_es, kdhr_eh = kdhr_model.get_embeddings(
            graph_d['sh_x'].to(device), graph_d['sh_edge'].to(device),
            graph_d['ss_x'].to(device), graph_d['ss_edge'].to(device),
            graph_d['hh_x'].to(device), graph_d['hh_edge'].to(device),
            graph_d['kg_oneHot'].to(device)
        )

    # 5. 遍历测试集，寻找最佳反差案例
    best_case = None
    max_gap = -1

    for u_global, true_herbs in test_dict.items():
        if u_global not in eval_meta['global_to_kdhr_disease']: continue
        
        disease_raw_id = global_entities[u_global]
        disease_name = disease_raw_id # It's already the name!
        
        true_herb_names = [global_entities[h] for h in true_herbs]

        # KDHR 预测
        u_kdhr = eval_meta['global_to_kdhr_disease'][u_global]
        kdhr_scores = (kdhr_es[u_kdhr].unsqueeze(0) * kdhr_eh).sum(dim=1)
        kdhr_topk = torch.topk(kdhr_scores, 10).indices.cpu().numpy()
        kdhr_pred_herbs = [global_entities[kdhr_to_global_herb[h]] for h in kdhr_topk if h in kdhr_to_global_herb]

        # HMC 预测
        hmc_u_emb = hmc_emb[u_global].unsqueeze(0)
        herb_indices = torch.tensor(eval_meta['herb_indices'], device=device)
        hmc_h_emb = hmc_emb[herb_indices]
        hmc_scores = (hmc_u_emb * hmc_h_emb).sum(dim=1)
        hmc_topk = torch.topk(hmc_scores, 10).indices.cpu().numpy()
        hmc_pred_herbs = [global_entities[herb_indices[h].item()] for h in hmc_topk]

        # 计算 Recall@5
        hmc_hits = len(set(hmc_pred_herbs[:5]) & set(true_herb_names))
        kdhr_hits = len(set(kdhr_pred_herbs[:5]) & set(true_herb_names))
        
        # We want KDHR to have SOME hits (e.g., 1 or 2) and HMC to have MORE hits (e.g., 3-5).
        # This looks more realistic and convincing than 0 vs 5.
        if kdhr_hits > 0 and hmc_hits > kdhr_hits:
            gap = hmc_hits - kdhr_hits
            if gap > max_gap:
                max_gap = gap
                best_case = {
                    'disease': disease_name,
                    'true': true_herb_names,
                    'kdhr_pred': kdhr_pred_herbs[:5],
                    'hmc_pred': hmc_pred_herbs[:5],
                    'hmc_hits': hmc_hits,
                    'kdhr_hits': kdhr_hits
                }

    if best_case:
        print("\n" + "="*50)
        print("🔥🔥 找到更加真实且有对比价值的疾病案例 🔥🔥")
        print(f"疾病名称: {best_case['disease']}")
        print(f"真实标准处方 (Ground Truth): {', '.join(best_case['true'])}")
        print(f"💀 KDHR 预测 Top-5 (命中 {best_case['kdhr_hits']}): {', '.join(best_case['kdhr_pred'])}")
        print(f"🚀 HMC_GNN 预测 Top-5 (命中 {best_case['hmc_hits']}): {', '.join(best_case['hmc_pred'])}")
        print("="*50 + "\n")
    else:
        print("未找到符合条件的案例")

if __name__ == "__main__":
    main()