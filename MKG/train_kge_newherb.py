# train_kge_newherb.py (Fixed: Offline Fusion Version)

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.nn import MarginRankingLoss
import numpy as np
import pandas as pd
import os
from tqdm import tqdm
from collections import defaultdict
from pathlib import Path
import random

# =================================================================
# 1. 基础组件
# =================================================================
class KGDataset(Dataset):
    def __init__(self, df, ent2ix, rel2ix):
        self.head_idx = torch.tensor([ent2ix[h] for h in df['from']], dtype=torch.long)
        self.tail_idx = torch.tensor([ent2ix[t] for t in df['to']], dtype=torch.long)
        self.relations = torch.tensor([rel2ix[r] for r in df['rel']], dtype=torch.long)
    def __len__(self): return len(self.head_idx)
    def __getitem__(self, idx): return self.head_idx[idx], self.tail_idx[idx], self.relations[idx]

@torch.no_grad()
def self_adversarial_negative_sampler(h_batch, t_batch, r_batch, model, num_entities, device, temperature, num_candidates, mode):
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
    else: # mode == 'head'
        neg_candidates = torch.randint(0, num_entities, (batch_size, num_candidates), device=device)
        t_emb = model.get_entity_embeddings(t_batch).unsqueeze(1)
        r_emb = model.rel_emb(r_batch).unsqueeze(1)
        neg_cand_emb = model.get_entity_embeddings(neg_candidates.view(-1)).view(batch_size, num_candidates, -1)
        
        # RotatE Head Prediction: || h - (t * r^-1) ||
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
# 2. 模型定义: ModalRotatE (简化版 - 仅接收融合后的特征)
# =================================================================
class ModalRotatE(nn.Module):
    def __init__(self, text_embeddings, n_relations, emb_dim, dropout_p=0.0):
        super().__init__()
        # ✅ 修正：只接收 text_embeddings (这里实际上是 final_fused_embeddings)
        # 不再需要 smiles_embeddings, component_indices_map, gate_layer
        self.text_ent_emb = nn.Embedding.from_pretrained(torch.from_numpy(text_embeddings).float(), freeze=False)
        
        self.rel_dim = emb_dim // 2
        self.rel_emb = nn.Embedding(n_relations, self.rel_dim)
        
        emb_range = 6.0 / np.sqrt(self.rel_dim)
        nn.init.uniform_(tensor=self.rel_emb.weight.data, a=-emb_range, b=emb_range)
        
        self.output_dropout = nn.Dropout(dropout_p)

    def get_entity_embeddings(self, indices):
        # 直接返回，因为融合已经在外部完成了
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
# 3. 评估函数 (带过滤 & 类型约束)
# =================================================================
@torch.no_grad()
def evaluate_kge_typed(model, dataset, all_triples_map, relation2types, type2entity_ids, ix2rel, device, batch_size=128):
    model.eval()
    ranks = []
    
    all_entity_ids = torch.arange(model.text_ent_emb.num_embeddings).to(device)
    all_entity_embs_fused = model.get_entity_embeddings(all_entity_ids)
    
    dataloader = DataLoader(dataset, batch_size=batch_size)
    
    for h_idx, t_idx, r_idx in tqdm(dataloader, desc="Evaluating"):
        h_idx, t_idx, r_idx = h_idx.to(device), t_idx.to(device), r_idx.to(device)
        
        h_emb = model.get_entity_embeddings(h_idx)
        r_emb = model.rel_emb(r_idx)
        scores = model._scoring(h_emb, r_emb, all_entity_embs_fused)
        
        for i in range(h_idx.shape[0]):
            h, r, t = h_idx[i].item(), r_idx[i].item(), t_idx[i].item()
            
            rel_name = ix2rel[r]
            if rel_name in relation2types:
                tail_type = relation2types[rel_name][1]
                if tail_type in type2entity_ids:
                    valid_candidates = type2entity_ids[tail_type]
                    type_mask = torch.ones(scores.shape[1], dtype=torch.bool, device=device)
                    type_mask[valid_candidates] = False
                    scores[i, type_mask] = float('inf')

            filter_ids = all_triples_map.get((h, r), [])
            filter_mask = [f_id for f_id in filter_ids if f_id != t]
            if filter_mask:
                scores[i, filter_mask] = float('inf')
        
        _, sorted_indices = torch.sort(scores, dim=1, descending=False)
        for i in range(h_idx.shape[0]):
            try:
                rank = (sorted_indices[i] == t_idx[i]).nonzero(as_tuple=True)[0].item() + 1
                ranks.append(rank)
            except:
                ranks.append(model.text_ent_emb.num_embeddings)

    model.train()
    ranks = np.array(ranks)
    return {"mrr": (1. / ranks).mean(), "hits@1": (ranks <= 1).mean(), "hits@10": (ranks <= 10).mean()}

# =================================================================
# 4. 主逻辑
# =================================================================
def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # --- 路径 ---
    SCRIPT_DIR = Path(__file__).parent.resolve()
    DATA_ROOT = SCRIPT_DIR / "dataset" / "NEWHERB"
    KGE_DIR = DATA_ROOT / "kge_data"
    FEATURE_DIR = SCRIPT_DIR / "output" / "NEWHERB"
    OUTPUT_DIR = SCRIPT_DIR / "output" / "newherb_model"
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 1. 加载数据
    print("Loading datasets...")
    df_train = pd.read_csv(KGE_DIR / 'train.tsv', sep='\t', header=None, names=['from', 'rel', 'to'])
    df_val = pd.read_csv(KGE_DIR / 'dev.tsv', sep='\t', header=None, names=['from', 'rel', 'to'])
    df_test = pd.read_csv(KGE_DIR / 'test.tsv', sep='\t', header=None, names=['from', 'rel', 'to'])
    
    with open(KGE_DIR / "entities.txt", 'r', encoding='utf-8') as f:
        all_entities = [l.strip() for l in f if l.strip()]
    ent2ix = {e: i for i, e in enumerate(all_entities)}
    
    with open(KGE_DIR / "relations.txt", 'r', encoding='utf-8') as f:
        all_relations = [l.strip() for l in f if l.strip()]
    rel2ix = {r: i for i, r in enumerate(all_relations)}
    
    ix2rel = {i: r for r, i in rel2ix.items()}
    n_entities = len(all_entities)
    n_relations = len(all_relations)
    print(f"Entities: {n_entities}, Relations: {n_relations}")

    # 2. 加载类型约束
    print("Loading type constraints...")
    entity2type = {p[0]: p[1] for p in (l.strip().split('\t') for l in open(KGE_DIR / "entity2type.txt", 'r', encoding='utf-8')) if len(p) == 2}
    relation2types = {p[0]: (p[1], p[2]) for p in (l.strip().split('\t') for l in open(KGE_DIR / "relation2types.txt", 'r', encoding='utf-8')) if len(p) == 3}
    
    type2entity_ids = defaultdict(list)
    for ent, etype in entity2type.items():
        if ent in ent2ix:
            type2entity_ids[etype].append(ent2ix[ent])
    
    type2entity_ids_gpu = {k: torch.tensor(v, dtype=torch.long, device=device) for k, v in type2entity_ids.items()}

    # 3. 加载最终特征
    print("Loading final embeddings...")
    emb_path = FEATURE_DIR / "final_entity_embeddings_text_smiles.npy"
    if not emb_path.exists(): raise FileNotFoundError(f"Missing {emb_path}")
    final_embeddings = np.load(emb_path)

    # 4. 准备训练
    train_dataset = KGDataset(df_train, ent2ix, rel2ix)
    val_dataset = KGDataset(df_val, ent2ix, rel2ix)
    test_dataset = KGDataset(df_test, ent2ix, rel2ix)
    
    all_triples_map = defaultdict(set)
    for df in [df_train, df_val, df_test]:
        for _, row in df.iterrows():
            if row['from'] in ent2ix and row['to'] in ent2ix:
                h, r, t = ent2ix[row['from']], rel2ix[row['rel']], ent2ix[row['to']]
                all_triples_map[(h, r)].add(t)
    all_triples_map = {k: list(v) for k, v in all_triples_map.items()}

    # 5. 初始化模型 (✅ 修正参数调用)
    model = ModalRotatE(
        text_embeddings=final_embeddings, 
        n_relations=n_relations,
        emb_dim=final_embeddings.shape[1],
        dropout_p=0.0
    )
    
    criterion = MarginRankingLoss(margin=1.0)
    model.to(device)
    criterion.to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    
    train_dataloader = DataLoader(train_dataset, batch_size=512, shuffle=True, num_workers=4)

    # 6. 训练循环
    n_epochs = 500
    best_mrr = 0.0
    patience = 30          # 连续 30 次验证没有提升就停
    no_improve = 0
    adv_temp = 1.0
    n_neg = 64

    print(f"\nStarting training on NEWHERB...")
    for epoch in range(n_epochs):
        model.train()
        total_loss = 0
        
        # -------------------------
        # Train for one epoch
        # -------------------------
        for h, t, r in tqdm(train_dataloader, desc=f"Epoch {epoch+1}"):
            h, t, r = h.to(device), t.to(device), r.to(device)
            optimizer.zero_grad()

            # Tail prediction
            n_t = self_adversarial_negative_sampler(
                h, t, r, model, n_entities, device, adv_temp, n_neg, 'tail'
            )
            pos_t, neg_t = model(h, r, t, None, n_t, 'tail')
            loss_t = criterion(pos_t, neg_t, torch.tensor([-1.0], device=device))

            # Head prediction
            n_h = self_adversarial_negative_sampler(
                h, t, r, model, n_entities, device, adv_temp, n_neg, 'head'
            )
            pos_h, neg_h = model(h, r, t, n_h, None, 'head')
            loss_h = criterion(pos_h, neg_h, torch.tensor([-1.0], device=device))

            loss = (loss_t + loss_h) / 2
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            # Normalize entity embeddings
            with torch.no_grad():
                model.text_ent_emb.weight.data = F.normalize(
                    model.text_ent_emb.weight.data, p=2, dim=1
                )

            total_loss += loss.item()

        avg_loss = total_loss / len(train_dataloader)
        print(f"Epoch {epoch+1} Loss: {avg_loss:.6f}")

        # ======================================
        # Validation Every Epoch  (Your Requirement)
        # ======================================
        print("Validating...")
        metrics = evaluate_kge_typed(
            model, val_dataset, all_triples_map,
            relation2types, type2entity_ids_gpu,
            ix2rel, device
        )
        print(f"Val Metrics: {metrics}")

        mrr = metrics["mrr"]

        # -------------------------
        # Best model check
        # -------------------------
        if mrr > best_mrr:
            best_mrr = mrr
            no_improve = 0
            torch.save(model.state_dict(), OUTPUT_DIR / "best_model.pt")
            print(f"⭐ New Best MRR: {best_mrr:.6f} (model saved!)")
        else:
            no_improve += 1
            print(f"⚠ No improvement: {no_improve}/{patience}")

        # -------------------------
        # Early stopping trigger
        # -------------------------
        if no_improve >= patience:
            print("⏹ Early stopping triggered (30 validations with no improvement).")
            break

    print("\n=== Final Testing ===")
    model.load_state_dict(torch.load(OUTPUT_DIR / "best_model.pt"))
    test_metrics = evaluate_kge_typed(model, test_dataset, all_triples_map, relation2types, type2entity_ids_gpu, ix2rel, device)
    print(f"Test Results: {test_metrics}")

if __name__ == "__main__":
    main()