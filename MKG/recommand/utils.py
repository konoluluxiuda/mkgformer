import torch
import numpy as np
import random
import os
from config import Config

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

class Evaluator:
    def __init__(self, k_list=None):
        self.k_list = k_list if k_list else Config.top_k

    def evaluate(self, model, test_dict, all_herb_ids, edge_index, edge_type):
        """
        执行 Top-K 评估 (适用于 Unified Graph / HMC_GNN_SSL)
        test_dict: {disease_idx: [ground_truth_herb_indices]}
        """
        model.eval()
        device = Config.device
        
        metrics = {k: {'p': [], 'r': [], 'f1': []} for k in self.k_list}
        
        # 1. 获取 inference 模式下的节点嵌入 (关闭扰动)
        with torch.no_grad():
            full_emb = model.forward_encoder(edge_index, edge_type, perturbed=False)
            
            # 准备候选药材的 Embedding 矩阵 [Num_Herbs, Dim]
            candidate_tensor = torch.tensor(list(all_herb_ids), dtype=torch.long, device=device)
            herb_embs = full_emb[candidate_tensor]
            
            # 2. 遍历测试集
            for disease_idx, truth_list in test_dict.items():
                if len(truth_list) == 0: continue
                
                # 获取当前疾病 Embedding [1, Dim]
                u_emb = full_emb[disease_idx].unsqueeze(0)
                
                # =========================================================
                # 关键一致性保证：使用纯内积计算分数 (与 train.py 的 BPR 对应)
                # =========================================================
                # [1, Dim] @ [Dim, Num_Herbs] -> [1, Num_Herbs] -> squeeze -> [Num_Herbs]
                scores = torch.matmul(u_emb, herb_embs.t()).squeeze() 
                
                # 获取 Top-MaxK 索引 
                max_k = max(self.k_list)
                _, top_indices = torch.topk(scores, k=max_k)
                
                # 将下标转换回全局 Entity ID
                top_global_ids = candidate_tensor[top_indices].cpu().numpy()
                
                # 3. 计算指标
                for k in self.k_list:
                    rec_k = set(top_global_ids[:k])
                    truth_set = set(truth_list)
                    
                    hits = len(rec_k & truth_set)
                    
                    p = hits / k
                    r = hits / len(truth_set)
                    f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
                    
                    metrics[k]['p'].append(p)
                    metrics[k]['r'].append(r)
                    metrics[k]['f1'].append(f1)
        
        # 4. 汇总结果
        results = {}
        for k in self.k_list:
            # 论文中通常报告百分比 (或小数保留4位)
            results[f'Precision@{k}'] = np.mean(metrics[k]['p'])
            results[f'Recall@{k}'] = np.mean(metrics[k]['r'])
            results[f'F1@{k}'] = np.mean(metrics[k]['f1'])
            
        return results

class SplitEvaluator:
    """
    专门用于物理拆分图模型 (MultiView_GNN) 的评估器
    输入模型接收 graphs 字典，而不是 edge_index 张量
    """
    def __init__(self, k_list=[5, 10, 20]):
        self.k_list = k_list

    def evaluate(self, model, graphs, test_dict, all_herb_ids, device):
        model.eval()
        
        metrics = {k: {'p': [], 'r': [], 'f1': []} for k in self.k_list}
        
        with torch.no_grad():
            final_emb = model.forward_encoder(graphs, perturbed=False)
            
            candidate_tensor = torch.tensor(list(all_herb_ids), dtype=torch.long, device=device)
            herb_embs = final_emb[candidate_tensor] 
            
            for disease_idx, truth_list in test_dict.items():
                if len(truth_list) == 0: continue
                
                u_emb = final_emb[disease_idx].unsqueeze(0)
                
                # =========================================================
                # 注意：你之前这里的代码是 model.predict_score()
                # 如果你在 model_split.py 里删除了 predict_score，这里会报错。
                # 为了安全，这里也统一改为纯内积。
                # =========================================================
                scores = torch.matmul(u_emb, herb_embs.t()).squeeze()
                
                max_k = max(self.k_list)
                _, top_indices = torch.topk(scores, k=max_k)
                
                top_global_ids = candidate_tensor[top_indices].cpu().numpy()
                truth_set = set(truth_list)
                
                for k in self.k_list:
                    rec_k = set(top_global_ids[:k])
                    
                    hits = len(rec_k & truth_set)
                    
                    p = hits / k
                    r = hits / len(truth_set)
                    f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
                    
                    metrics[k]['p'].append(p)
                    metrics[k]['r'].append(r)
                    metrics[k]['f1'].append(f1)
        
        results = {}
        for k in self.k_list:
            results[f'Precision@{k}'] = np.mean(metrics[k]['p'])
            results[f'Recall@{k}'] = np.mean(metrics[k]['r'])
            results[f'F1@{k}'] = np.mean(metrics[k]['f1'])
            
        return results