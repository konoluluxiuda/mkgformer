import os
import sys
import torch
import numpy as np
from tqdm import tqdm
import random

# 导入本地模块
from config import Config
from dataset import GraphDataManager
from utils import set_seed

from bsgam_model import BSGAMAdapted
from bsgam_wrapper import BSGAMWrapper
from train_bsgam_newherb import load_bsgam_data, BSGAM_EMB_DIM, BSGAM_HEAD_NUM, BSGAM_ATT_DROP, BSGAM_KG_DIM

from kdhr_wrapper import KDHRWrapper
from train_kdhr_newherb import load_kdhr_data, KDHR_EMB_DIM, KDHR_BATCH, KDHR_DROP, KDHR_KG_DIM

# 添加项目根目录到路径以便导入 KDHR
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
MKG_DIR = os.path.dirname(CURRENT_DIR)
PROJECT_ROOT = os.path.dirname(MKG_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
from KDHR.model import KDHR

from tcmpr_model_adapted import TCMPRAdapted
from tcmpr_wrapper import TCMPRWrapper
from train_tcmpr_newherb import load_tcmpr_data, MAX_SYMPTOM_NUM, CONV_FILTERS, KERNEL_SIZE, FUSION, LAYER1, LAYER2, EMBED_DIM, DROPOUT

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
BSGAM_CKPT = os.path.join(CURRENT_DIR, 'checkpoints', 'bsgam_best.pt')
KDHR_CKPT = os.path.join(CURRENT_DIR, 'checkpoints', 'kdhr_best.pt')
TCMPR_CKPT = os.path.join(CURRENT_DIR, 'checkpoints', 'tcmpr_best.pt')

def evaluate_long_tail(x, train_dict, test_dict, herb_indices, head_herbs, mid_herbs, tail_herbs, K=10):
    group_hits = {'head': 0, 'mid': 0, 'tail': 0}
    group_gt = {'head': 0, 'mid': 0, 'tail': 0}
    
    all_herbs_tensor = torch.tensor(herb_indices, dtype=torch.long, device=Config.device)
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
        herb_id_to_idx = {h: i for i, h in enumerate(herb_indices)}
        train_idx = [herb_id_to_idx[h] for h in train_h if h in herb_id_to_idx]
        if train_idx:
            scores[train_idx] = -1e9
            
        # Get Top-K
        _, topk_idx = torch.topk(scores, K)
        topk_herbs = [herb_indices[i] for i in topk_idx.cpu().numpy()]
        
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
                continue
                
            group_gt[group] += 1
            if h in topk_herbs:
                group_hits[group] += 1

    head_recall = group_hits['head'] / group_gt['head'] if group_gt['head'] > 0 else 0
    mid_recall = group_hits['mid'] / group_gt['mid'] if group_gt['mid'] > 0 else 0
    tail_recall = group_hits['tail'] / group_gt['tail'] if group_gt['tail'] > 0 else 0
    
    print(f"  Head (Top-20%) Recall@{K}: {head_recall:.4f}  ({group_hits['head']} hits / {group_gt['head']} total GT)")
    print(f"  Mid  (Mid-40%) Recall@{K}: {mid_recall:.4f}  ({group_hits['mid']} hits / {group_gt['mid']} total GT)")
    print(f"  Tail (Btm-40%) Recall@{K}: {tail_recall:.4f}  ({group_hits['tail']} hits / {group_gt['tail']} total GT)")
    print("-" * 50)
    return head_recall, mid_recall, tail_recall


def main():
    print("=" * 50)
    print("Long-tail Distribution Evaluation (Baselines)")
    print("=" * 50)

    set_seed(Config.seed)
    device = torch.device(Config.device)
    
    # 强制读取 Paper Graph 来获取同样的 train/test 数据划分
    Config.REC_DATA_DIR = os.path.join(Config.DATA_ROOT, 'paper_graph_data')
    data_manager = GraphDataManager()
    _, _, train_dict, test_dict = data_manager.load_data()

    # 复刻测试集一半划分
    all_test_users = list(test_dict.keys())
    all_test_users.sort() 
    random.seed(Config.seed)
    random.shuffle(all_test_users)
    half_idx = len(all_test_users) // 2
    new_test_dict = {u: test_dict[u] for u in all_test_users[half_idx:]}
    test_dict = new_test_dict 

    herb_indices = data_manager.herb_indices
    
    # 统计出现频率
    herb_counts = {h: 0 for h in herb_indices}
    for u, herbs in train_dict.items():
        for h in herbs:
            if h in herb_counts:
                herb_counts[h] += 1
                
    sorted_herbs = sorted(herb_counts.items(), key=lambda x: x[1], reverse=True)
    num_herbs = len(sorted_herbs)
    head_cutoff = int(num_herbs * 0.2)
    mid_cutoff = int(num_herbs * 0.6)
    
    head_herbs = set([x[0] for x in sorted_herbs[:head_cutoff]])
    mid_herbs = set([x[0] for x in sorted_herbs[head_cutoff:mid_cutoff]])
    tail_herbs = set([x[0] for x in sorted_herbs[mid_cutoff:]])
    
    eval_dict = {
        'train_dict': train_dict,
        'test_dict': test_dict,
        'herb_indices': herb_indices,
        'head_herbs': head_herbs,
        'mid_herbs': mid_herbs,
        'tail_herbs': tail_herbs
    }

    # dummy edge needed for signatures
    dummy_edge = torch.zeros(2, 0, dtype=torch.long, device=device)
    dummy_type = torch.zeros(0, dtype=torch.long, device=device)
    
    # ==========================
    # 1. BSGAM Evaluation
    # ==========================
    print("\n>>> Evaluating BSGAM")
    try:
        bsgam_data = load_bsgam_data(device)
        model_bsgam = BSGAMAdapted(
            num_diseases=bsgam_data['num_diseases'],
            num_herbs=bsgam_data['num_herbs'],
            input_dim=bsgam_data['input_dim'],
            embedding_dim=BSGAM_EMB_DIM,
            head_num=BSGAM_HEAD_NUM,
            att_drop=BSGAM_ATT_DROP,
            kg_dim=BSGAM_KG_DIM,
        ).to(device)
        model_bsgam.load_state_dict(torch.load(BSGAM_CKPT, map_location=device))
        model_bsgam.eval()
        
        wrapper_bsgam = BSGAMWrapper(model_bsgam, bsgam_data['graph_data'], bsgam_data['eval_meta'], device=device)
        
        with torch.no_grad():
            x = wrapper_bsgam.forward_encoder(dummy_edge, dummy_type)
            evaluate_long_tail(x, **eval_dict)
    except Exception as e:
        print(f"BSGAM Error: {e}")

    # ==========================
    # 2. KDHR Evaluation
    # ==========================
    print("\n>>> Evaluating KDHR")
    try:
        kdhr_data = load_kdhr_data()
        kg_dim = kdhr_data['eval_meta'].get('kg_dim', KDHR_KG_DIM)
        num_diseases = kdhr_data['num_diseases']
        num_herbs = kdhr_data['num_herbs']
        sh_num = num_diseases + num_herbs
        model_kdhr = KDHR(
            num_diseases, num_herbs, sh_num,
            KDHR_EMB_DIM, KDHR_BATCH, KDHR_DROP, kg_dim=kg_dim
        ).to(device)
        model_kdhr.load_state_dict(torch.load(KDHR_CKPT, map_location=device))
        model_kdhr.eval()
        
        wrapper_kdhr = KDHRWrapper(model_kdhr, kdhr_data['graph_data'], kdhr_data['eval_meta'], device=device)
        with torch.no_grad():
            x = wrapper_kdhr.forward_encoder(dummy_edge, dummy_type)
            evaluate_long_tail(x, **eval_dict)
    except Exception as e:
        print(f"KDHR Error: {e}")

    # ==========================
    # 3. TCMPR Evaluation
    # ==========================
    print("\n>>> Evaluating TCMPR")
    try:
        tcmpr_data = load_tcmpr_data(device)
        model_tcmpr = TCMPRAdapted(
            symptom_dim=tcmpr_data['symptom_dim'],
            herb_count=tcmpr_data['num_herbs'],
            max_symptom_num=MAX_SYMPTOM_NUM,
            conv_filters=CONV_FILTERS,
            kernel_size=KERNEL_SIZE,
            fusion=FUSION,
            layer1=LAYER1,
            layer2=LAYER2,
            embed_dim=EMBED_DIM,
            dropout=DROPOUT,
        ).to(device)
        model_tcmpr.load_state_dict(torch.load(TCMPR_CKPT, map_location=device))
        model_tcmpr.eval()
        
        # tcmpr_wrapper expect `symptom_seq_all` which is tcmpr_data['symptom_seq_all']
        wrapper_tcmpr = TCMPRWrapper(model_tcmpr, tcmpr_data['symptom_seq_all'], tcmpr_data['eval_meta'], device=device)
        with torch.no_grad():
            x = wrapper_tcmpr.forward_encoder(dummy_edge, dummy_type)
            evaluate_long_tail(x, **eval_dict)
    except Exception as e:
        print(f"TCMPR Error: {e}")

if __name__ == "__main__":
    main()