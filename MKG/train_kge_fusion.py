# train_kge_fusion.py (Final Self-Contained Version)
# train_kge_fusion.py	ModalEx	动态 (可学习门控)	torchkge 评估器 (有过滤, 无类型约束)  使用的是complex
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import pandas as pd
import os

from torchkge.sampling import UniformNegativeSampler
# ✅ 1. 导入正确的 BCEWithLogitsLoss
from torch.nn import BCEWithLogitsLoss
from torchkge.utils import DataLoader
from torchkge.data_structures import KnowledgeGraph
from torchkge.evaluation import LinkPredictionEvaluator

# =================================================================
# 1. 定义我们自己的、完全独立的模型
# =================================================================
class ModalEx(nn.Module):
    def __init__(self, text_embeddings, smiles_embeddings, component_indices_map, n_relations, emb_dim):
        super().__init__()
        
        self.text_ent_emb = nn.Embedding.from_pretrained(torch.from_numpy(text_embeddings).float(), freeze=False)
        self.smiles_ent_emb = nn.Embedding.from_pretrained(torch.from_numpy(smiles_embeddings).float(), freeze=False)
        self.rel_emb = nn.Embedding(n_relations, emb_dim)
        nn.init.xavier_uniform_(self.rel_emb.weight.data)
        
        self.component_indices_map = component_indices_map
        
        self.gate_layer = nn.Sequential(
            nn.Linear(emb_dim * 2, emb_dim // 4),
            nn.ReLU(),
            nn.Linear(emb_dim // 4, 1),
            nn.Sigmoid()
        )

    def get_entity_embeddings(self, indices):
        final_embeddings = self.text_ent_emb(indices)
        mask = torch.tensor(
            [idx.item() in self.component_indices_map for idx in indices], 
            dtype=torch.bool, 
            device=indices.device
        )
        component_locs = torch.where(mask)[0]
        if component_locs.numel() > 0:
            component_ids = indices[component_locs]
            text_emb_comps = self.text_ent_emb(component_ids)
            smiles_emb_comps = self.smiles_ent_emb(component_ids)
            gate = self.gate_layer(torch.cat([text_emb_comps, smiles_emb_comps], dim=1))
            fused_embeddings = gate * text_emb_comps + (1 - gate) * smiles_emb_comps
            final_embeddings[component_locs] = fused_embeddings
        return final_embeddings
    
    # _scoring 方法现在只负责计算 ComplEx 的核心数学运算
    def _scoring(self, h_emb, r_emb, t_emb):
        # 分离所有输入的实部和虚部
        h_re, h_im = h_emb.chunk(2, dim=1)
        r_re, r_im = r_emb.chunk(2, dim=1)
        t_re, t_im = t_emb.chunk(2, dim=1)

        # Case 1: 尾实体预测 (Tail Prediction)
        # h 和 r 是 batch, t 是所有实体
        # h/r: [b, d], t: [N, d] -> output: [b, N]
        if t_emb.shape[0] > h_emb.shape[0]:
            hr_re = h_re * r_re - h_im * r_im
            hr_im = h_re * r_im + h_im * r_re
            return (hr_re @ t_re.transpose(0, 1)) + (hr_im @ t_im.transpose(0, 1))

        # Case 2: 头实体预测 (Head Prediction)
        # t 和 r 是 batch, h 是所有实体
        # t/r: [b, d], h: [N, d] -> output: [b, N]
        elif h_emb.shape[0] > t_emb.shape[0]:
            # ComplEx score for (h,r,t) is <h_re, r_re, t_re> + <h_re, r_im, t_im> + ...
            # To predict h, we compute scores for all h's.
            # This is equivalent to <r_re, t_re, h_re> + <r_im, t_im, h_re> + ...
            rt_re = r_re * t_re + r_im * t_im
            rt_im = r_re * t_im - r_im * t_re
            return (rt_re @ h_re.transpose(0, 1)) + (rt_im @ h_im.transpose(0, 1))

        # Case 3: 训练 (Training)
        # h, r, t 都是 batch, 形状相同
        # h/r/t: [b, d] -> output: [b]
        else:
            return torch.sum(h_re * r_re * t_re + h_re * r_im * t_im + h_im * r_re * t_im - h_im * r_im * t_re, -1)
    
    # inference_scoring_function 现在只是一个简单的包装器
    def inference_scoring_function(self, h, t, r):
        return self._scoring(h_emb=h, r_emb=r, t_emb=t)

    def inference_prepare_candidates(self, h_idx, t_idx, r_idx, entities=True):
        # ... (这个方法不变)
        h_emb = self.get_entity_embeddings(h_idx)
        r_emb = self.rel_emb(r_idx)
        candidates = self.get_entity_embeddings(torch.arange(self.text_ent_emb.num_embeddings, device=h_emb.device))
        t_emb = self.get_entity_embeddings(t_idx)
        return h_emb, t_emb, r_emb, candidates

    def forward(self, h_idx, t_idx, r_idx, n_h_idx, n_t_idx):
        # forward 方法只用于训练
        h_emb = self.get_entity_embeddings(h_idx)
        t_emb = self.get_entity_embeddings(t_idx)
        r_emb = self.rel_emb(r_idx)
        n_h_emb = self.get_entity_embeddings(n_h_idx)
        n_t_emb = self.get_entity_embeddings(n_t_idx)
        
        score_pos = self._scoring(h_emb, r_emb, t_emb)
        score_neg_h = self._scoring(n_h_emb, r_emb, t_emb)
        score_neg_t = self._scoring(h_emb, r_emb, n_t_emb)
        
        return score_pos, torch.cat([score_neg_h, score_neg_t])
# =================================================================
# 2. 主脚本逻辑
# =================================================================
def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # --- 路径配置 ---
    data_dir = "dataset/HERB"
    feature_dir = "output/HERB"
    output_dir = "output/text_smiles_model" # 新的、更清晰的输出目录
    os.makedirs(output_dir, exist_ok=True)

    # --- 加载数据和特征 ---
    with open(os.path.join(data_dir, "entities.txt"), 'r', encoding='utf-8') as f:
        full_entity_list = [line.strip() for line in f if line.strip()]
    full_entity_map = {name: i for i, name in enumerate(full_entity_list)}

    text_embeddings = np.load(os.path.join(feature_dir, "entity_embeddings_text_only.npy"))
    smiles_embeddings_raw = np.load(os.path.join(feature_dir, "component_smiles_embeddings.npy"))
    with open(os.path.join(feature_dir, "component_smiles_map.txt"), 'r', encoding='utf-8') as f:
        smiles_component_list = [line.strip() for line in f]

    # --- 对齐 Embedding 矩阵 ---
    smiles_embeddings = np.zeros_like(text_embeddings)
    component_indices_map = {}

    print("Aligning SMILES embeddings with full entity list...")
    for i, comp_name in enumerate(smiles_component_list):
        if comp_name in full_entity_map:
            entity_id = full_entity_map[comp_name]
            smiles_embeddings[entity_id] = smiles_embeddings_raw[i]
            component_indices_map[entity_id] = True

    df_train = pd.read_csv(os.path.join(data_dir, 'train.tsv'), sep='\t', header=None, names=['from', 'rel', 'to'])
    df_val = pd.read_csv(os.path.join(data_dir, 'dev.tsv'), sep='\t', header=None, names=['from', 'rel', 'to'])
    df_test = pd.read_csv(os.path.join(data_dir, 'test.tsv'), sep='\t', header=None, names=['from', 'rel', 'to'])
    df_all = pd.concat([df_train, df_val, df_test], ignore_index=True)
    kg_all = KnowledgeGraph(df=df_all)
    kg_train = KnowledgeGraph(df=df_train, ent2ix=kg_all.ent2ix, rel2ix=kg_all.rel2ix)
    kg_val = KnowledgeGraph(df=df_val, ent2ix=kg_all.ent2ix, rel2ix=kg_all.rel2ix)
    kg_test = KnowledgeGraph(df=df_test, ent2ix=kg_all.ent2ix, rel2ix=kg_all.rel2ix)

    # --- 初始化我们自己的 ModalEx 模型 ---
    model = ModalEx(
        emb_dim=text_embeddings.shape[1],
        n_relations=kg_all.n_rel,
        text_embeddings=text_embeddings,
        smiles_embeddings=smiles_embeddings,
        component_indices_map=component_indices_map,
    )
    # ✅ 核心修复 1: 使用 BCEWithLogitsLoss
    criterion = nn.BCEWithLogitsLoss()
    model.to(device)
    criterion.to(device)

    # --- 训练循环 (可以进行超参数调优) ---
    train_dataloader = DataLoader(kg_train, batch_size=512) # 尝试更大的 batch size
    optimizer = optim.Adam(model.parameters(), lr=1e-4) # 尝试一个不同的学习率
    sampler = UniformNegativeSampler(kg_train)
    n_epochs = 300 # 增加训练轮数
    
    print(f"\nStarting training for {n_epochs} epochs with ModalEx...")
    for epoch in range(n_epochs):
        running_loss = 0.0
        for i, batch in enumerate(train_dataloader):
            h, t, r = batch[0].to(device), batch[1].to(device), batch[2].to(device)
            n_h, n_t = sampler.corrupt_batch(h, t, r)
            optimizer.zero_grad()
            scores_pos, scores_neg = model(h_idx=h, t_idx=t, r_idx=r, n_h_idx=n_h, n_t_idx=n_t)
            scores = torch.cat([scores_pos, scores_neg], dim=0)
            targets = torch.cat([torch.ones_like(scores_pos), torch.zeros_like(scores_neg)], dim=0)
            loss = criterion(scores, targets)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        epoch_loss = running_loss / len(train_dataloader)
        print(f'Epoch {epoch+1}/{n_epochs}, Loss: {epoch_loss:.6f}')

    # =================================================================
    # ✅ 3. 加入评估和预测
    # =================================================================
    torch.save(model.state_dict(), os.path.join(output_dir, "modalex_model.pt"))
    print(f"\nTraining complete. Model saved to {os.path.join(output_dir, 'modalex_model.pt')}")

    print("\nEvaluating model on validation set...")
    evaluator_val = LinkPredictionEvaluator(model=model, knowledge_graph=kg_val)
    evaluator_val.evaluate(b_size=128) # Use a smaller batch size for evaluation if memory is an issue
    print("\n--- Validation Set Evaluation Results ---")
    evaluator_val.print_results()

    print("\nEvaluating model on test set...")
    evaluator_test = LinkPredictionEvaluator(model=model, knowledge_graph=kg_test)
    evaluator_test.evaluate(b_size=128)
    print("\n--- Test Set Evaluation Results ---")
    evaluator_test.print_results()
    
    # =================================================================
    # 6. 使用模型进行预测的示例
    # =================================================================
    print("\n--- Prediction Example ---")
    try:
        h_str = '三七'
        r_str = 'belongs_to_meridian'
        ix2ent = {v: k for k, v in kg_all.ent2ix.items()}
        h_idx = torch.tensor([kg_all.ent2ix[h_str]]).to(device)
        r_idx = torch.tensor([kg_all.rel2ix[r_str]]).to(device)
        
        # ✅ 核心修复：调用正确的预测接口
        # 1. 准备评估所需的 embedding
        h_emb = model.get_entity_embeddings(h_idx)
        r_emb = model.rel_emb(r_idx)
        all_entities_emb = model.get_entity_embeddings(
            torch.arange(model.text_ent_emb.num_embeddings).to(device)
        )
        
        # 2. 调用专门为评估设计的评分函数
        scores = model.inference_scoring_function(h=h_emb, t=all_entities_emb, r=r_emb).squeeze(0)
        
        # 3. 后续逻辑不变
        top_k_scores, top_k_indices = torch.topk(scores, k=10, largest=True)
        top_k_entities = [ix2ent[idx.item()] for idx in top_k_indices]

        print(f"Top 10 predicted meridian tropism for '{h_str}':")
        for entity, score in zip(top_k_entities, top_k_scores):
            print(f"  - Entity: {entity}, Score (similarity): {score.item():.4f}")
    except KeyError as e:
        print(f"Could not run prediction example: {e}")

if __name__ == "__main__":
    main()