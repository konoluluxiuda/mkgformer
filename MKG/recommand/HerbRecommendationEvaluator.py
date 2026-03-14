import numpy as np
import torch

class HerbRecommendationEvaluator:
    def __init__(self, k_list=[5, 10, 20]):
        """
        初始化评估器
        :param k_list: 需要评估的 K 值列表，例如 [5, 10, 20]
        """
        self.k_list = k_list

    def calculate_metrics(self, recommended_items, ground_truth_items):
        """
        核心计算函数，对应论文公式 (21)-(23)
        :param recommended_items: 模型预测出的 Top-K 药材 ID 列表 (对应公式中的 K)
        :param ground_truth_items: 真实的药材 ID 集合 (对应公式中的 h_set)
        """
        # 转化为集合以便计算交集
        rec_set = set(recommended_items)
        truth_set = set(ground_truth_items)
        
        # 计算交集 |h_set ∩ K|
        intersection = len(rec_set & truth_set)
        
        # 公式 (21): Precision@K = |h_set ∩ K| / |K|
        # 注意：这里的 |K| 就是 len(recommended_items)
        precision = intersection / len(recommended_items) if len(recommended_items) > 0 else 0.0
        
        # 公式 (22): Recall@K = |h_set ∩ K| / |h_set|
        recall = intersection / len(truth_set) if len(truth_set) > 0 else 0.0
        
        # 公式 (23): F1-Score@K
        if (precision + recall) > 0:
            f1 = (2 * precision * recall) / (precision + recall)
        else:
            f1 = 0.0
            
        return precision, recall, f1

    def evaluate(self, model, test_dataloader, all_herb_ids, device):
        """
        批量评估函数
        :param model: 训练好的 GNN 模型 (需实现 predict 或 return scores)
        :param test_dataloader: 测试集加载器
        :param all_herb_ids: 所有候选药材的 ID 列表 (用于映射索引)
        :param device: CPU/GPU
        """
        model.eval()
        
        # 存储每个 K 值的累积结果
        metrics_sum = {k: {'p': 0.0, 'r': 0.0, 'f1': 0.0} for k in self.k_list}
        num_samples = 0
        
        all_candidates_tensor = torch.LongTensor(all_herb_ids).to(device)

        print(f"开始评估... (测试集样本数: {len(test_dataloader.dataset)})")
        
        with torch.no_grad():
            for batch in test_dataloader:
                # 假设 Batch 数据结构: 
                # inputs: 疾病/症状 ID (Batch Size)
                # targets: 真实药材列表 (List of Lists, 因为每个病人的药方长度不同)
                inputs = batch['input_id'].to(device)
                ground_truths = batch['ground_truth_herbs'] # 这是一个列表的列表
                
                # 1. 模型预测
                # Paper B 公式 (1): P(H|S_set) = f(S_set, theta)
                # 输出 scores 维度: [Batch_Size, Num_All_Herbs]
                scores = model.predict(inputs, all_candidates_tensor)
                
                # 2. 遍历 Batch 中的每一个样本
                for i in range(len(inputs)):
                    truth_ids = ground_truths[i] # 当前病人的真实药方 (h_set)
                    
                    # 过滤掉空数据的样本
                    if len(truth_ids) == 0:
                        continue
                        
                    user_scores = scores[i] # 当前病人的预测分数
                    
                    # 3. 对不同的 K 进行评估
                    max_k = max(self.k_list)
                    
                    # 获取分数最高的 Top-MaxK 的索引
                    _, top_indices = torch.topk(user_scores, k=max_k)
                    top_herb_ids = [all_herb_ids[idx] for idx in top_indices.cpu().numpy()]
                    
                    for k in self.k_list:
                        # 截取前 K 个推荐结果 (对应公式中的 K 集合)
                        k_recs = top_herb_ids[:k]
                        
                        # 计算单样本指标
                        p, r, f1 = self.calculate_metrics(k_recs, truth_ids)
                        
                        # 累加
                        metrics_sum[k]['p'] += p
                        metrics_sum[k]['r'] += r
                        metrics_sum[k]['f1'] += f1
                    
                    num_samples += 1

        # 4. 计算平均值
        final_results = {}
        for k in self.k_list:
            final_results[f'Precision@{k}'] = metrics_sum[k]['p'] / num_samples
            final_results[f'Recall@{k}']    = metrics_sum[k]['r'] / num_samples
            final_results[f'F1@{k}']        = metrics_sum[k]['f1'] / num_samples
            
        return final_results