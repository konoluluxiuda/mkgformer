# search_hyperparams.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
import os
from tqdm import tqdm
from collections import defaultdict
from pathlib import Path
import json

# =================================================================
# 1. 导入我们已经写好的模型、数据集和评估函数
#    (将 train_kge_rotate_manual.py 的核心组件复制过来)
# =================================================================
class KGDataset(Dataset):
    def __init__(self, df, ent2ix, rel2ix):
        self.head_idx = torch.tensor([ent2ix[h] for h in df['from']], dtype=torch.long)
        self.tail_idx = torch.tensor([ent2ix[t] for t in df['to']], dtype=torch.long)
        self.relations = torch.tensor([rel2ix[r] for r in df['rel']], dtype=torch.long)
    def __len__(self): return len(self.head_idx)
    def __getitem__(self, idx): return self.head_idx[idx], self.tail_idx[idx], self.relations[idx]

def uniform_negative_sampler(h_batch, t_batch, num_entities, device):
    batch_size = h_batch.shape[0]
    corrupt_head_mask = torch.rand(batch_size, device=device) > 0.5
    neg_indices = torch.randint(0, num_entities, (batch_size,), device=device)
    n_h = torch.where(corrupt_head_mask, neg_indices, h_batch)
    n_t = torch.where(~corrupt_head_mask, neg_indices, t_batch)
    return n_h, n_t


class ModalRotatE(nn.Module):
    def __init__(self, text_embeddings, smiles_embeddings, component_indices_map, n_relations, emb_dim):
        super().__init__()
        self.text_ent_emb = nn.Embedding.from_pretrained(torch.from_numpy(text_embeddings).float(), freeze=False)
        self.smiles_ent_emb = nn.Embedding.from_pretrained(torch.from_numpy(smiles_embeddings).float(), freeze=False)
        self.rel_dim = emb_dim // 2
        self.rel_emb = nn.Embedding(n_relations, self.rel_dim)
        
        emb_range = 6.0 / np.sqrt(self.rel_dim)
        nn.init.uniform_(tensor=self.rel_emb.weight.data, a=-emb_range, b=emb_range)
        
        self.component_indices_map = component_indices_map
        self.gate_layer = nn.Sequential(
            nn.Linear(emb_dim * 2, emb_dim // 4),
            nn.ReLU(),
            nn.Linear(emb_dim // 4, 1),
            nn.Sigmoid()
        )

    def get_entity_embeddings(self, indices):
        final_embeddings = self.text_ent_emb(indices)
        mask = torch.tensor([idx.item() in self.component_indices_map for idx in indices], dtype=torch.bool, device=indices.device)
        component_locs = torch.where(mask)[0]
        if component_locs.numel() > 0:
            component_ids = indices[component_locs]
            text_emb_comps = self.text_ent_emb(component_ids)
            smiles_emb_comps = self.smiles_ent_emb(component_ids)
            gate = self.gate_layer(torch.cat([text_emb_comps, smiles_emb_comps], dim=1))
            fused_embeddings = gate * text_emb_comps + (1 - gate) * smiles_emb_comps
            final_embeddings[component_locs] = fused_embeddings
        return final_embeddings
    
    def _scoring(self, h_emb, r_emb, t_emb):
        h_re, h_im = h_emb.chunk(2, dim=1)
        phase_r = r_emb / (6.0 / np.sqrt(self.rel_dim) / np.pi)
        r_re, r_im = torch.cos(phase_r), torch.sin(phase_r)
        
        if h_emb.shape[0] != t_emb.shape[0]: # 评估模式
            t_re, t_im = t_emb.chunk(2, dim=1)
            h_re, h_im = h_re.unsqueeze(1), h_im.unsqueeze(1)
            r_re, r_im = r_re.unsqueeze(1), r_im.unsqueeze(1)
        else: # 训练模式
            t_re, t_im = t_emb.chunk(2, dim=1)
        
        re_score = (h_re * r_re - h_im * r_im) - t_re
        im_score = (h_re * r_im + h_im * r_re) - t_im
        return torch.sqrt(re_score**2 + im_score**2).sum(dim=-1)

    def forward(self, h_idx, r_idx, t_idx=None, n_idx=None, mode='tail'):
        h_emb = self.get_entity_embeddings(h_idx)
        r_emb = self.rel_emb(r_idx)
        
        if t_idx is not None and n_idx is not None:
            t_emb = self.get_entity_embeddings(t_idx)
            n_emb = self.get_entity_embeddings(n_idx)
            score_pos = self._scoring(h_emb, r_emb, t_emb)
            if mode == 'tail':
                score_neg = self._scoring(h_emb, r_emb, n_emb)
            else:
                score_neg = self._scoring(n_emb, r_emb, t_emb)
            return score_pos, score_neg
        else:
            all_entities_emb = self.get_entity_embeddings(torch.arange(self.text_ent_emb.num_embeddings).to(h_emb.device))
            return self._scoring(h_emb, r_emb, all_entities_emb)

@torch.no_grad()
def evaluate_kge(model, dataset, all_triples_map, device, batch_size=128):
    model.eval()
    ranks = []
    all_entity_ids = torch.arange(model.text_ent_emb.num_embeddings).to(device)
    
    # 评估时，我们一次性计算好所有融合后的实体 embedding，以提高效率
    all_entity_embs_fused = model.get_entity_embeddings(all_entity_ids)
    
    for h_idx, t_idx, r_idx in tqdm(DataLoader(dataset, batch_size=batch_size), desc="Evaluating"):
        h_idx, t_idx, r_idx = h_idx.to(device), t_idx.to(device), r_idx.to(device)
        
        # 尾实体预测
        h_emb = model.get_entity_embeddings(h_idx)
        r_emb = model.rel_emb(r_idx)
        scores = model._scoring(h_emb, r_emb, all_entity_embs_fused)
        
        for i in range(h_idx.shape[0]):
            h, r, t = h_idx[i].item(), r_idx[i].item(), t_idx[i].item()
            filter_ids = all_triples_map.get((h, r), [])
            scores[i, [f_id for f_id in filter_ids if f_id != t]] = float('inf')
        
        _, sorted_indices = torch.sort(scores, dim=1, descending=False)
        for i in range(h_idx.shape[0]):
            rank = (sorted_indices[i] == t_idx[i]).nonzero(as_tuple=True)[0].item() + 1
            ranks.append(rank)

    model.train()
    ranks = np.array(ranks)
    return {"mrr": (1. / ranks).mean(), "hits@1": (ranks <= 1).mean(), "hits@10": (ranks <= 10).mean()}


# =================================================================
# 2. 定义训练一个模型的函数
# =================================================================
def train_one_model(params, train_dataset, val_dataset, text_embeddings, smiles_embeddings, 
                    component_indices_map, n_entities, n_relations, all_triples_map, device):
    """
    使用一组给定的超参数，训练一个模型并返回验证集上的 MRR。
    """
    print("\n" + "="*50)
    print(f"Testing parameters: {params}")
    print("="*50)

    # --- 初始化模型 ---
    model = ModalRotatE(
        emb_dim=text_embeddings.shape[1],
        n_relations=n_relations,
        text_embeddings=text_embeddings,
        smiles_embeddings=smiles_embeddings,
        component_indices_map=component_indices_map,
    )
    
    criterion = nn.MarginRankingLoss(margin=params['margin'])
    model.to(device)
    criterion.to(device)
    optimizer = optim.Adam(model.parameters(), lr=params['lr'], weight_decay=params['weight_decay'])
    
    train_dataloader = DataLoader(train_dataset, batch_size=params['batch_size'], shuffle=True, num_workers=4, pin_memory=True)
    
    # --- 训练循环 (较短的 epoch) ---
    for epoch in range(params['n_epochs']):
        model.train()
        running_loss = 0.0
        # 内部循环不显示进度条，以保持日志整洁
        for h, t, r in train_dataloader:
            h, t, r = h.to(device), t.to(device), r.to(device)
            optimizer.zero_grad()
            n_h, n_t = uniform_negative_sampler(h, t, n_entities, device)
            
            pos_scores_t, neg_scores_t = model(h_idx=h, r_idx=r, t_idx=t, n_idx=n_t, mode='tail')
            loss_t = criterion(pos_scores_t, neg_scores_t, torch.tensor([-1], device=device))
            
            pos_scores_h, neg_scores_h = model(h_idx=h, r_idx=r, t_idx=t, n_idx=n_h, mode='head')
            loss_h = criterion(pos_scores_h, neg_scores_h, torch.tensor([-1], device=device))
            
            loss = (loss_h + loss_t) / 2.0
            loss.backward()
            optimizer.step()
            
            with torch.no_grad():
                model.text_ent_emb.weight.data = F.normalize(model.text_ent_emb.weight.data, p=2, dim=1)
                model.smiles_ent_emb.weight.data = F.normalize(model.smiles_ent_emb.weight.data, p=2, dim=1)
            
            running_loss += loss.item()
        
        epoch_loss = running_loss / len(train_dataloader)
        print(f'Epoch {epoch+1}/{params["n_epochs"]}, Loss: {epoch_loss:.6f}')

    # --- 评估 ---
    evaluation_batch_size = 64 # 从一个小值开始尝试
    print(f"Starting evaluation with batch_size = {evaluation_batch_size}...")
    val_metrics = evaluate_kge(model, val_dataset, all_triples_map, device, batch_size=evaluation_batch_size)
    print(f"Validation MRR for params {params}: {val_metrics['mrr']:.6f}")
    
    return val_metrics['mrr']


# =================================================================
# 3. 主搜索逻辑
# =================================================================
def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Starting hyperparameter search on device: {device}")
    
    # --- 路径配置 ---
    data_dir = Path("dataset/HERB")
    feature_dir = Path("output/HERB")
    output_dir = Path("output/rotate_model")
    output_dir.mkdir(exist_ok=True, parents=True)

    # --- 加载数据 ---
    df_train = pd.read_csv(data_dir / 'train.tsv', sep='\t', header=None, names=['from', 'rel', 'to'])
    df_val = pd.read_csv(data_dir / 'dev.tsv', sep='\t', header=None, names=['from', 'rel', 'to'])
    df_test = pd.read_csv(data_dir / 'test.tsv', sep='\t', header=None, names=['from', 'rel', 'to'])
    
    # --- 创建实体和关系映射 ---
    print("Building entity and relation mappings...")
    all_entities_list = sorted(list(set(df_train['from']) | set(df_train['to']) | set(df_val['from']) | set(df_val['to']) | set(df_test['from']) | set(df_test['to'])))
    all_relations_list = sorted(list(set(df_train['rel']) | set(df_val['rel']) | set(df_test['rel'])))
    ent2ix = {ent: i for i, ent in enumerate(all_entities_list)}
    rel2ix = {rel: i for i, rel in enumerate(all_relations_list)}
    ix2ent = {i: ent for ent, i in ent2ix.items()}
    n_entities = len(all_entities_list)
    n_relations = len(all_relations_list)
    print(f"Found {n_entities} unique entities and {n_relations} unique relations.")

    # --- 加载和对齐特征 ---
    print("Loading and aligning pre-trained embeddings...")
    text_embeddings_raw = np.load(feature_dir / "entity_embeddings_text_only.npy")
    with open(Path(data_dir) / "entities.txt", 'r', encoding='utf-8') as f:
        # 这个列表的顺序是原始特征的顺序
        text_entity_list = [line.strip() for line in f if line.strip()]
    text_entity_map = {name: i for i, name in enumerate(text_entity_list)}

    smiles_embeddings_raw = np.load(feature_dir / "component_smiles_embeddings.npy")
    with open(feature_dir / "component_smiles_map.txt", 'r', encoding='utf-8') as f:
        smiles_component_list = [line.strip() for line in f]
    smiles_component_map = {name: i for i, name in enumerate(smiles_component_list)}

    # 创建空的 embedding 矩阵，准备填充
    text_embeddings = np.zeros((n_entities, text_embeddings_raw.shape[1]))
    smiles_embeddings = np.zeros_like(text_embeddings)
    component_indices_map = {}

    for i, entity_name in enumerate(all_entities_list):
        # 根据我们新创建的权威 ent2ix 顺序 (i)，填充 embedding
        if entity_name in text_entity_map:
            text_embeddings[i] = text_embeddings_raw[text_entity_map[entity_name]]
        else:
            print(f"Warning: Entity '{entity_name}' from TSV not found in text embedding map.")
        
        if entity_name in smiles_component_map:
            smiles_embeddings[i] = smiles_embeddings_raw[smiles_component_map[entity_name]]
            component_indices_map[i] = True

    # --- 创建数据集和 DataLoader ---
    train_dataset = KGDataset(df_train, ent2ix, rel2ix)
    val_dataset = KGDataset(df_val, ent2ix, rel2ix)
    test_dataset = KGDataset(df_test, ent2ix, rel2ix)
    train_dataloader = DataLoader(train_dataset, batch_size=512, shuffle=True, num_workers=4, pin_memory=True)
            
    # --- 创建过滤 map ---
    all_triples_map = defaultdict(list)
    for df in [df_train, df_val, df_test]:
        for _, row in df.iterrows():
            if row['from'] in ent2ix and row['rel'] in rel2ix and row['to'] in ent2ix:
                h, r, t = ent2ix[row['from']], rel2ix[row['rel']], ent2ix[row['to']]
                all_triples_map[(h, r)].append(t)
            
    # --- 初始化模型 ---
    model = ModalRotatE(
        emb_dim=text_embeddings.shape[1],
        n_relations=n_relations,
        text_embeddings=text_embeddings,
        smiles_embeddings=smiles_embeddings,
        component_indices_map=component_indices_map,
    )

    # =============================================================
    # ✅ 核心：定义超参数搜索空间
    # =============================================================
    param_grid = {
        'lr': [5e-5, 1e-4, 2e-4],          # 3个学习率: 保持核心探索范围
        'margin': [1.0],                  # 1个margin: 暂时固定在我们已知效果不错的 1.0 上
        'batch_size': [256, 512],         # 2个批次大小: 对比标准和较大批次
        'weight_decay': [0.0, 1e-5]       # 2个权重衰减: 对比无正则化和标准正则化
    }
    
    # 设定一个合理的搜索轮数
    search_epochs = 200
    
    results = []
    best_mrr = -1
    best_params = None

    # --- 遍历所有参数组合 ---
    # (这是一个简单的网格搜索)
    for lr in param_grid['lr']:
        for margin in param_grid['margin']:
            for batch_size in param_grid['batch_size']:
                for weight_decay in param_grid['weight_decay']:
                    
                    current_params = {
                        'lr': lr,
                        'margin': margin,
                        'batch_size': batch_size,
                        'weight_decay': weight_decay,
                        'n_epochs': search_epochs
                    }
                    
                    # 训练并评估一个模型
                    val_mrr = train_one_model(
                        params=current_params,
                        train_dataset=train_dataset,
                        val_dataset=val_dataset,
                        text_embeddings=text_embeddings,
                        smiles_embeddings=smiles_embeddings,
                        component_indices_map=component_indices_map,
                        n_entities=n_entities,
                        n_relations=n_relations,
                        all_triples_map=all_triples_map,
                        device=device
                    )

                    results.append({'params': current_params, 'mrr': val_mrr})

                    if val_mrr > best_mrr:
                        best_mrr = val_mrr
                        best_params = current_params

    # --- 报告最佳结果 ---
    print("\n" + "#"*50)
    print("Hyperparameter search complete!")
    print(f"Best validation MRR: {best_mrr:.6f}")
    print(f"Best parameters found: {best_params}")
    print("#"*50)

    # 将所有结果保存到文件
    with open("hyperparameter_search_results.json", "w") as f:
        json.dump(results, f, indent=4)
    print("All search results saved to hyperparameter_search_results.json")


if __name__ == "__main__":
    main()



