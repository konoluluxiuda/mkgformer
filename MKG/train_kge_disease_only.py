import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, TensorDataset
from torch.nn import MarginRankingLoss
import numpy as np
import pandas as pd
import os
from tqdm import tqdm
from collections import defaultdict
from pathlib import Path
import random

# =================================================================
# 1. 基础组件 (不变)
# =================================================================
class KGDataset(Dataset):
    def __init__(self, df, ent2ix, rel2ix):
        # 确保传入的 df 已经是经过筛选的
        self.head_idx = torch.tensor([ent2ix[h] for h in df['from']], dtype=torch.long)
        self.tail_idx = torch.tensor([ent2ix[t] for t in df['to']], dtype=torch.long)
        self.relations = torch.tensor([rel2ix[r] for r in df['rel']], dtype=torch.long)
    def __len__(self): return len(self.head_idx)
    def __getitem__(self, idx): return self.head_idx[idx], self.tail_idx[idx], self.relations[idx]

@torch.no_grad()
def self_adversarial_negative_sampler(h_batch, t_batch, r_batch, model, num_entities, device, temperature, num_candidates, mode):
    # (保持原有的自对抗采样代码不变)
    batch_size = h_batch.shape[0]
    if mode == 'tail':
        neg_candidates = torch.randint(0, num_entities, (batch_size, num_candidates), device=device)
        h_emb = model.get_entity_embeddings(h_batch).unsqueeze(1)
        r_emb = model.rel_emb(r_batch).unsqueeze(1)
        neg_cand_emb = model.get_entity_embeddings(neg_candidates.view(-1)).view(batch_size, num_candidates, -1)
        neg_scores = -model._scoring(h_emb, r_emb, neg_cand_emb)
        sampling_probs = F.softmax(neg_scores * temperature, dim=1)
        final_neg_indices_local = torch.multinomial(sampling_probs, 1).squeeze(1)
        n_t = neg_candidates[torch.arange(batch_size), final_neg_indices_local]
        return n_t
    else: 
        neg_candidates = torch.randint(0, num_entities, (batch_size, num_candidates), device=device)
        t_emb = model.get_entity_embeddings(t_batch).unsqueeze(1)
        r_emb = model.rel_emb(r_batch).unsqueeze(1)
        neg_cand_emb = model.get_entity_embeddings(neg_candidates.view(-1)).view(batch_size, num_candidates, -1)
        phase_r = r_emb / (6.0 / np.sqrt(model.rel_dim) / np.pi)
        r_re, r_im = torch.cos(phase_r), torch.sin(phase_r)
        t_re, t_im = t_emb.chunk(2, dim=-1)
        tr_inv_re = t_re * r_re + t_im * r_im
        tr_inv_im = t_im * r_re - t_re * r_im
        h_re, h_im = neg_cand_emb.chunk(2, dim=-1)
        re_score = h_re - tr_inv_re
        im_score = h_im - tr_inv_im
        neg_scores = -torch.sqrt(re_score**2 + im_score**2 + 1e-12).sum(dim=-1)
        sampling_probs = F.softmax(neg_scores * temperature, dim=1)
        final_neg_indices_local = torch.multinomial(sampling_probs, 1).squeeze(1)
        n_h = neg_candidates[torch.arange(batch_size), final_neg_indices_local]
        return n_h

# =================================================================
# 2. 模型定义: ModalRotatE (不变)
# =================================================================
class ModalRotatE(nn.Module):
    def __init__(self, text_embeddings, n_relations, emb_dim, dropout_p=0.0):
        super().__init__()
        self.text_ent_emb = nn.Embedding.from_pretrained(torch.from_numpy(text_embeddings).float(), freeze=False)
        self.rel_dim = emb_dim // 2
        self.rel_emb = nn.Embedding(n_relations, self.rel_dim)
        emb_range = 6.0 / np.sqrt(self.rel_dim)
        nn.init.uniform_(tensor=self.rel_emb.weight.data, a=-emb_range, b=emb_range)
        self.output_dropout = nn.Dropout(dropout_p)

    def get_entity_embeddings(self, indices):
        emb = self.text_ent_emb(indices)
        return self.output_dropout(emb)
    
    def _scoring(self, h_emb, r_emb, t_emb):
        h_re, h_im = h_emb.chunk(2, dim=-1)
        phase_r = r_emb / (6.0 / np.sqrt(self.rel_dim) / np.pi)
        r_re, r_im = torch.cos(phase_r), torch.sin(phase_r)
        
        if h_emb.dim() != t_emb.dim() or h_emb.shape[0] != t_emb.shape[0]:
            if t_emb.dim() == 2 and h_emb.dim() == 2:
                 t_re, t_im = t_emb.chunk(2, dim=-1)
                 h_re, h_im = h_re.unsqueeze(1), h_im.unsqueeze(1)
                 r_re, r_im = r_re.unsqueeze(1), r_im.unsqueeze(1)
            else:
                 t_re, t_im = t_emb.chunk(2, dim=-1)
            re_score = (h_re * r_re - h_im * r_im) - t_re
            im_score = (h_re * r_im + h_im * r_re) - t_im
            return torch.sqrt(re_score**2 + im_score**2 + 1e-12).sum(dim=-1)
        else:
            t_re, t_im = t_emb.chunk(2, dim=-1)
            re_score = (h_re * r_re - h_im * r_im) - t_re
            im_score = (h_re * r_im + h_im * r_re) - t_im
            return torch.sqrt(re_score**2 + im_score**2 + 1e-12).sum(dim=-1)

    def forward(self, h_idx, r_idx, t_idx=None, n_h_idx=None, n_t_idx=None, mode='tail'):
        h_emb = self.get_entity_embeddings(h_idx)
        r_emb = self.rel_emb(r_idx)
        
        if t_idx is not None:
            t_emb = self.get_entity_embeddings(t_idx)
            if mode == 'tail':
                n_emb = self.get_entity_embeddings(n_t_idx)
                score_pos = self._scoring(h_emb, r_emb, t_emb)
                score_neg = self._scoring(h_emb, r_emb, n_emb)
            else:
                n_emb = self.get_entity_embeddings(n_h_idx)
                score_pos = self._scoring(h_emb, r_emb, t_emb)
                score_neg = self._scoring(n_emb, r_emb, t_emb)
            return score_pos, score_neg
        else:
            all_entities_emb = self.get_entity_embeddings(torch.arange(self.text_ent_emb.num_embeddings).to(h_emb.device))
            return self._scoring(h_emb, r_emb, all_entities_emb)

# =================================================================
# 3. 疾病预测评估函数 (不变)
# =================================================================
@torch.no_grad()
def evaluate_disease_prediction(model, dataset, all_triples_map, target_rel_id, target_candidate_ids, device, batch_size=128):
    model.eval()
    ranks = []
    candidate_embs = model.get_entity_embeddings(target_candidate_ids)
    
    # 优化：构建 map
    cand_list = target_candidate_ids.cpu().tolist()
    global2local = {gid: idx for idx, gid in enumerate(cand_list)}
    
    dataloader = DataLoader(dataset, batch_size=batch_size)
    
    for h_idx, t_idx, r_idx in tqdm(dataloader, desc="Eval Diseases"):
        h_idx, t_idx, r_idx = h_idx.to(device), t_idx.to(device), r_idx.to(device)
        mask = (r_idx == target_rel_id)
        if not mask.any(): continue
            
        h_batch, t_batch, r_batch = h_idx[mask], t_idx[mask], r_idx[mask]
        
        h_emb = model.get_entity_embeddings(h_batch)
        r_emb = model.rel_emb(r_batch)
        scores = model._scoring(h_emb, r_emb, candidate_embs)
        
        t_batch_list = t_batch.cpu().tolist()
        h_batch_list = h_batch.cpu().tolist()
        r_batch_list = r_batch.cpu().tolist()
        
        for i in range(len(h_batch_list)):
            h_val, r_val, t_val = h_batch_list[i], r_batch_list[i], t_batch_list[i]
            
            if t_val in global2local:
                target_local_idx = global2local[t_val]
                filter_ids = all_triples_map.get((h_val, r_val), [])
                mask_indices = [global2local[fid] for fid in filter_ids if fid != t_val and fid in global2local]
                
                if mask_indices: scores[i, mask_indices] = float('inf')
                
                _, sorted_indices = torch.sort(scores[i], descending=False)
                rank = (sorted_indices == target_local_idx).nonzero().item() + 1
                ranks.append(rank)

    model.train()
    if not ranks: return {"mrr": 0.0, "hits@1": 0.0, "hits@10": 0.0, "count": 0}
    ranks = np.array(ranks)
    return {"mrr": (1. / ranks).mean(), "hits@1": (ranks <= 1).mean(), "hits@10": (ranks <= 10).mean(), "count": len(ranks)}

# =================================================================
# 4. 主逻辑
# =================================================================
def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    SCRIPT_DIR = Path(__file__).parent.resolve()
    DATA_ROOT = SCRIPT_DIR / "dataset" / "NEWHERB"
    KGE_DIR = DATA_ROOT / "kge_data"
    FEATURE_DIR = SCRIPT_DIR / "output" / "NEWHERB"
    OUTPUT_DIR = SCRIPT_DIR / "output" / "herb_disease_model_only" # 新的输出目录
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 1. 加载数据
    print("Loading datasets...")
    df_train = pd.read_csv(KGE_DIR / 'train.tsv', sep='\t', header=None, names=['from', 'rel', 'to'])
    df_val = pd.read_csv(KGE_DIR / 'dev.tsv', sep='\t', header=None, names=['from', 'rel', 'to'])
    df_test = pd.read_csv(KGE_DIR / 'test.tsv', sep='\t', header=None, names=['from', 'rel', 'to'])
    
    # 2. 映射构建
    with open(KGE_DIR / "entities.txt") as f:
        all_entities = [l.strip() for l in f]
    ent2ix = {e: i for i, e in enumerate(all_entities)}
    
    with open(KGE_DIR / "relations.txt") as f:
        all_relations = [l.strip() for l in f]
    rel2ix = {r: i for i, r in enumerate(all_relations)}
    
    n_entities = len(all_entities)
    n_relations = len(all_relations)

    # 3. 准备 Disease 相关信息
    target_rel_name = "treats_disease"
    if target_rel_name not in rel2ix:
        # 自动查找可能的疾病关系
        for r in rel2ix:
            if "disease" in r.lower(): target_rel_name = r; break
    target_rel_id = rel2ix[target_rel_name]
    print(f"Target Relation: {target_rel_name} ({target_rel_id})")
    
    entity2type = {p[0]: p[1] for p in (l.strip().split('\t') for l in open(KGE_DIR / "entity2type.txt")) if len(p)==2}
    disease_ids = [ent2ix[e] for e, t in entity2type.items() if t == 'Disease' and e in ent2ix]
    disease_ids_tensor = torch.tensor(disease_ids, dtype=torch.long, device=device)

    # =============================================================
    # 🔥 核心修改：过滤数据，只保留 Disease 相关三元组进行训练
    # =============================================================
    print("\n[FILTERING] Keeping ONLY Disease-related triples for training...")
    original_len = len(df_train)
    
    # 只保留关系为 treats_disease 的行
    df_train_filtered = df_train[df_train['rel'] == target_rel_name].copy()
    
    print(f"Original Training Triples: {original_len}")
    print(f"Filtered Training Triples (Disease Only): {len(df_train_filtered)}")
    print(f"Data Reduction: {(1 - len(df_train_filtered)/original_len)*100:.2f}% removed.")

    # 4. 过滤 Map (评估时还是建议用全量数据的过滤，防止把实际上正确的判错，或者也只用疾病过滤)
    # 这里为了公平对比，我们只记录 Disease 相关的过滤信息
    all_triples_map = defaultdict(set)
    # 合并所有数据用于过滤
    full_df = pd.concat([df_train, df_val, df_test])
    # 只保留 disease 关系 (可选，如果全量过滤更严谨)
    full_df = full_df[full_df['rel'] == target_rel_name]
    
    for _, row in full_df.iterrows():
        if row['from'] in ent2ix and row['to'] in ent2ix:
            h, r, t = ent2ix[row['from']], rel2ix[row['rel']], ent2ix[row['to']]
            all_triples_map[(h, r)].add(t)
    all_triples_map = {k: list(v) for k, v in all_triples_map.items()}

    # 5. 加载特征
    emb_path = FEATURE_DIR / "final_entity_embeddings_text_smiles.npy"
    final_embeddings = np.load(emb_path)

    # 6. 初始化模型
    model = ModalRotatE(
        text_embeddings=final_embeddings, 
        n_relations=n_relations,
        emb_dim=final_embeddings.shape[1],
        dropout_p=0.0 # 纯KGE通常不需要太大dropout
    ).to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    criterion = MarginRankingLoss(margin=1.0)
    
    # 7. 训练准备 (使用过滤后的数据)
    train_dataset = KGDataset(df_train_filtered, ent2ix, rel2ix)
    train_dataloader = DataLoader(train_dataset, batch_size=256, shuffle=True, num_workers=4) # 数据少了，batch可以小点

    val_dataset = KGDataset(df_val, ent2ix, rel2ix)
    test_dataset = KGDataset(df_test, ent2ix, rel2ix)

    # 8. 训练循环
    n_epochs = 300
    best_mrr = 0.0
    patience = 20
    no_improve = 0
    
    adv_temp = 1.0
    n_neg = 64

    print(f"\nStarting training (Disease Only)...")
    for epoch in range(n_epochs):
        model.train()
        total_loss = 0
        
        for h, t, r in tqdm(train_dataloader, desc=f"Epoch {epoch+1}"):
            h, t, r = h.to(device), t.to(device), r.to(device)
            optimizer.zero_grad()
            
            # 自对抗采样
            n_t = self_adversarial_negative_sampler(h, t, r, model, n_entities, device, adv_temp, n_neg, 'tail')
            pos_t, neg_t = model(h, r, t, None, n_t, 'tail')
            loss_t = criterion(pos_t, neg_t, torch.tensor([-1.0], device=device))
            
            n_h = self_adversarial_negative_sampler(h, t, r, model, n_entities, device, adv_temp, n_neg, 'head')
            pos_h, neg_h = model(h, r, t, n_h, None, 'head')
            loss_h = criterion(pos_h, neg_h, torch.tensor([-1.0], device=device))
            
            loss = (loss_t + loss_h) / 2
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            with torch.no_grad():
                model.text_ent_emb.weight.data = F.normalize(model.text_ent_emb.weight.data, p=2, dim=1)
                
            total_loss += loss.item()
            
        print(f"Epoch {epoch+1} Loss: {total_loss/len(train_dataloader):.6f}")

        # 验证
        if (epoch + 1) % 5 == 0:
            print("Validating...")
            metrics = evaluate_disease_prediction(
                model, val_dataset, all_triples_map, target_rel_id, disease_ids_tensor, device
            )
            print(f"Val Disease Metrics: {metrics}")
            
            if metrics.get('mrr', 0) > best_mrr:
                best_mrr = metrics['mrr']
                torch.save(model.state_dict(), OUTPUT_DIR / "best_model.pt")
                print("⭐ New Best Model Saved!")
                no_improve = 0
            else:
                no_improve += 5
                if no_improve >= patience:
                    print("Early stopping.")
                    break
    
    # 9. 最终测试
    print("\n=== Final Testing ===")
    if (OUTPUT_DIR / "best_model.pt").exists():
        model.load_state_dict(torch.load(OUTPUT_DIR / "best_model.pt"))
    
    test_metrics = evaluate_disease_prediction(
        model, test_dataset, all_triples_map, target_rel_id, disease_ids_tensor, device
    )
    print(f"Final Test Results: {test_metrics}")

if __name__ == "__main__":
    main()