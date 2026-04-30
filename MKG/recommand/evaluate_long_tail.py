import os
import torch
import numpy as np
from tqdm import tqdm
import random

# 导入本地模块
from config import Config
from dataset import GraphDataManager
from model import HMC_GNN_SSL
from utils import set_seed

def main():
    print("=" * 50)
    print("Long-tail Distribution Evaluation (HMC-GNN)")
    print("=" * 50)

    # 1. 强制设定与 train.py 相同的配置
    set_seed(Config.seed)
    FUSION_MODE = 'gated'
    USE_BASE_ATTR = True    
    USE_CHEM_DENSE = True
    USE_CROSS_MODAL = True           # 同步 train.py 开关
    USE_CHEM_FINGERPRINT = True      # 同步 train.py 开关
    USE_DISEASE_TEXT = True      # 同步 train.py 开关
    
    # 强制读取 Paper Graph
    Config.REC_DATA_DIR = os.path.join(Config.DATA_ROOT, 'paper_graph_data')
    data_manager = GraphDataManager()
    
    try:
        edge_index, edge_type, train_dict, test_dict = data_manager.load_data()
    except FileNotFoundError as e:
        print(f"Error loading graph data: {e}")
        return

    # 2. 复刻 train.py 的 Data Split 保证测试集 100% 对应
    all_test_users = list(test_dict.keys())
    all_test_users.sort() 
    random.seed(Config.seed)
    random.shuffle(all_test_users)
    
    half_idx = len(all_test_users) // 2
    new_test_dict = {}
    for u in all_test_users[half_idx:]:
        new_test_dict[u] = test_dict[u]
    test_dict = new_test_dict 

    # 3. 统计训练集中所有 herb 的出现频率 (Degree)
    herb_counts = {h: 0 for h in data_manager.herb_indices}
    for u, herbs in train_dict.items():
        for h in herbs:
            if h in herb_counts:
                herb_counts[h] += 1
                
    # 按照出现频次从大到小排序
    sorted_herbs = sorted(herb_counts.items(), key=lambda x: x[1], reverse=True)
    num_herbs = len(sorted_herbs)
    head_cutoff = int(num_herbs * 0.2)
    mid_cutoff = int(num_herbs * 0.6)
    
    head_herbs = set([x[0] for x in sorted_herbs[:head_cutoff]])
    mid_herbs = set([x[0] for x in sorted_herbs[head_cutoff:mid_cutoff]])
    tail_herbs = set([x[0] for x in sorted_herbs[mid_cutoff:]])
    
    print(f"\n[Herb Frequency Groups]")
    print(f"  Head (Top 20%):  {len(head_herbs)} herbs")
    print(f"  Mid  (20%-60%):  {len(mid_herbs)} herbs")
    print(f"  Tail (Bottom 40%): {len(tail_herbs)} herbs")

    # 4. 加载所有的多模态特征
    # Chem
    chem_matrix = None
    if USE_CROSS_MODAL:
        chem_path = os.path.join(Config.DATA_ROOT, 'recommendation_data', 'node_chem_dense.pt')
        if os.path.exists(chem_path):
            chem_matrix = torch.load(chem_path)
            if USE_CHEM_FINGERPRINT:
                fp_path_pt = os.path.join(Config.DATA_ROOT, 'recommendation_data', 'node_chem_fingerprint.pt')
                fp_path_npy = os.path.join(Config.DATA_ROOT, 'recommendation_data', 'node_chem_fingerprint.npy')
                fp_feat = None
                
                if os.path.exists(fp_path_pt):
                    fp_feat = torch.load(fp_path_pt)
                elif os.path.exists(fp_path_npy):
                    fp_feat = torch.from_numpy(np.load(fp_path_npy)).float()
                
                if fp_feat is not None:
                    chem_matrix = torch.cat([chem_matrix, fp_feat], dim=1)
                
            chem_matrix = chem_matrix.to(Config.device)

    # Attr
    attr_tensors = []
    if USE_BASE_ATTR:
        base_attr = data_manager.load_attributes()
        if base_attr is not None:
            attr_tensors.append(base_attr)
    
    if USE_CHEM_DENSE:
        chem_attr_path = os.path.join(Config.DATA_ROOT, 'recommendation_data', 'node_chem_multihot.pt')
        if os.path.exists(chem_attr_path):
            attr_tensors.append(torch.load(chem_attr_path))
    
    final_attr_matrix = torch.cat(attr_tensors, dim=1).to(Config.device) if attr_tensors else None

    # Disease
    disease_matrix = None
    if USE_DISEASE_TEXT:
        dis_path = os.path.join(Config.DATA_ROOT, 'recommendation_data', 'node_disease_text.pt')
        if os.path.exists(dis_path):
            disease_matrix = torch.load(dis_path).to(Config.device)

    # 5. 加载模型及权重
    model = HMC_GNN_SSL(
        num_nodes=data_manager.num_nodes,
        num_relations=data_manager.num_relations,
        pretrained_features=None,
        attr_matrix=final_attr_matrix,
        chem_matrix=chem_matrix,
        disease_matrix=disease_matrix,
        fusion_mode=FUSION_MODE
    ).to(Config.device)

    save_path = os.path.join(Config.MODEL_SAVE_PATH, 'best_model.pt')
    if os.path.exists(save_path):
        model.load_state_dict(torch.load(save_path, map_location=Config.device))
        print(f"\n✅ Loaded best model from {save_path}")
    else:
        print(f"\n❌ Error: Cannot find model weight {save_path}")
        return

    # 6. Evaluation
    model.eval()
    K = 10
    
    group_hits = {'head': 0, 'mid': 0, 'tail': 0}
    group_gt = {'head': 0, 'mid': 0, 'tail': 0}
    
    edge_index = edge_index.to(Config.device)
    edge_type = edge_type.to(Config.device)

    print(f"\nStarting Tail Evaluation for Recall@{K}...")
    with torch.no_grad():
        x = model.forward_encoder(edge_index, edge_type, perturbed=False)
        all_herbs_tensor = torch.tensor(data_manager.herb_indices, dtype=torch.long, device=Config.device)
        herb_emb = x[all_herbs_tensor] # [num_herbs, dim]
        
        for u, gt_herbs in tqdm(test_dict.items(), desc="Evaluating", leave=False):
            if not gt_herbs:
                continue
                
            u_t = torch.tensor([u], dtype=torch.long, device=Config.device)
            u_emb = x[u_t] # [1, dim]
            
            # Predict scores
            scores = torch.matmul(u_emb, herb_emb.t()).squeeze(0) # [num_herbs]
            
            # Filter out training items
            train_h = train_dict.get(u, [])
            herb_id_to_idx = {h: i for i, h in enumerate(data_manager.herb_indices)}
            train_idx = [herb_id_to_idx[h] for h in train_h if h in herb_id_to_idx]
            if train_idx:
                scores[train_idx] = -1e9
                
            # Get Top-K
            _, topk_idx = torch.topk(scores, K)
            topk_herbs = [data_manager.herb_indices[i] for i in topk_idx.cpu().numpy()]
            
            # Assign hits into buckets
            for h in gt_herbs:
                group = None
                if h in head_herbs:
                    group = 'head'
                elif h in mid_herbs:
                    group = 'mid'
                elif h in tail_herbs:
                    group = 'tail'
                else:
                    continue # Not in candidate list? Should not happen.
                    
                group_gt[group] += 1
                if h in topk_herbs:
                    group_hits[group] += 1

    print(f"\n=======================================================")
    print(f"  Recall@{K} Results based on Hub/Tail stratification")
    print(f"=======================================================")
    
    # 避免除以零
    head_recall = group_hits['head'] / group_gt['head'] if group_gt['head'] > 0 else 0
    mid_recall = group_hits['mid'] / group_gt['mid'] if group_gt['mid'] > 0 else 0
    tail_recall = group_hits['tail'] / group_gt['tail'] if group_gt['tail'] > 0 else 0
    
    print(f"  Head (Top-20%) Recall@{K}: {head_recall:.4f}  ({group_hits['head']} hits / {group_gt['head']} total GT)")
    print(f"  Mid  (Mid-40%) Recall@{K}: {mid_recall:.4f}  ({group_hits['mid']} hits / {group_gt['mid']} total GT)")
    print(f"  Tail (Btm-40%) Recall@{K}: {tail_recall:.4f}  ({group_hits['tail']} hits / {group_gt['tail']} total GT)")
    print(f"=======================================================")

if __name__ == "__main__":
    main()