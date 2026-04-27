import os
import torch
import numpy as np
import sys

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
MKG_DIR = os.path.dirname(CURRENT_DIR)
PROJECT_ROOT = os.path.dirname(MKG_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from config import Config
from dataset import GraphDataManager
from model import HMC_GNN_SSL

def main():
    device = torch.device(Config.device)
    
    # 1. Load Data
    data_manager = GraphDataManager()
    edge_index, edge_type, train_dict, test_dict = data_manager.load_data()
    mkg_dir = os.path.dirname(os.path.abspath(__file__))
    if 'recommand' in mkg_dir: mkg_dir = os.path.dirname(mkg_dir)
        
    with open(os.path.join(mkg_dir, 'dataset', 'NEWHERB', 'kge_data', 'entities.txt'), 'r') as f:
        global_entities = [line.strip() for line in f.readlines()]

    # 2. Target Diseases
    target_disease_names = [
        "Chronic Primary Insomnia",
        "Insomnia",
        "Asthma",
        "Asthma, Aspirin-Induced",
        "Atopic Asthma",
        "Cough",
        "Type 2 Diabetes",
        "Juvenile Rheumatoid Arthritis"
    ]

    # Find the global node ID for these targets
    target_ids = {}
    for idx, name in enumerate(global_entities):
        if name in target_disease_names:
            target_ids[name] = idx

    # 3. Load HMC_GNN_SSL Model
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

    # 4. Predict
    with torch.no_grad():
        hmc_emb = hmc_model.forward_encoder(edge_index.to(device), edge_type.to(device), perturbed=False)

        for disease_name, u_global in target_ids.items():
            print("\n" + "="*60)
            print(f"🩺 病症/疾病名称: {disease_name}")
            
            # Ground truth (if any in train or test)
            true_herbs = []
            if u_global in test_dict: true_herbs.extend(test_dict[u_global])
            if u_global in train_dict: true_herbs.extend(train_dict[u_global])
            
            # 去重：将 Train 和 Test 中的 Ground Truth 处方合并去重
            true_herb_names = list(set([global_entities[h] for h in true_herbs]))
            num_true_herbs = len(true_herb_names)
            
            # Extract scores
            hmc_u_emb = hmc_emb[u_global].unsqueeze(0)
            herb_indices = torch.tensor(data_manager.herb_indices, device=device)
            hmc_h_emb = hmc_emb[herb_indices]
            hmc_scores = (hmc_u_emb * hmc_h_emb).sum(dim=1)
            
            hmc_topk = torch.topk(hmc_scores, 10).indices.cpu().numpy()
            hmc_pred_herbs = [global_entities[herb_indices[h].item()] for h in hmc_topk]

            print(f"✅ Ground Truth 处方共包含 {num_true_herbs} 味中药: {', '.join(true_herb_names)[:100]}...")
            print(f"🚀 模型 Top-5 预测:  {', '.join(hmc_pred_herbs[:5])}")
            print(f"🚀 模型 Top-10 预测: {', '.join(hmc_pred_herbs)}")
            
            hits_5 = len(set(hmc_pred_herbs[:5]) & set(true_herb_names))
            hits_10 = len(set(hmc_pred_herbs) & set(true_herb_names))
            print(f"🎯 命中情况 - Top-5命中数: {hits_5}, Top-10命中数: {hits_10}")

            if num_true_herbs > 0:
                # Top-5 metrics
                p_5 = hits_5 / 5.0
                r_5 = hits_5 / float(num_true_herbs)
                f1_5 = 2.0 * (p_5 * r_5) / (p_5 + r_5) if (p_5 + r_5) > 0 else 0.0

                # Top-10 metrics
                p_10 = hits_10 / 10.0
                r_10 = hits_10 / float(num_true_herbs)
                f1_10 = 2.0 * (p_10 * r_10) / (p_10 + r_10) if (p_10 + r_10) > 0 else 0.0

                if disease_name == "Type 2 Diabetes":
                    print(f"📊 [Type 2 Diabetes计算结果] Precision@5: {p_5:.4f} | Recall@5: {r_5:.4f} | F1-score@5: {f1_5:.4f}")
                elif disease_name == "Cough":
                    print(f"📊 [Cough计算结果] Precision@10: {p_10:.4f} | Recall@10: {r_10:.4f} | F1-score@10: {f1_10:.4f}")
                else:
                    print(f"   Top-5  -> P: {p_5:.4f}, R: {r_5:.4f}, F1: {f1_5:.4f}")
                    print(f"   Top-10 -> P: {p_10:.4f}, R: {r_10:.4f}, F1: {f1_10:.4f}")
            else:
                print("⚠ 测试资料中无验方 Ground Truth 记录。")

if __name__ == "__main__":
    main()