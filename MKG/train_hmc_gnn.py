# train_hmc_gnn.py (Final Optimized Version)

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
import os
from tqdm import tqdm
from collections import defaultdict
from pathlib import Path

# PyG
from torch_geometric.nn import RGCNConv
from torch.nn import MarginRankingLoss

# =================================================================
# 1. 模型定义: HMC-GNN (RGCN Encoder + RotatE Decoder)
# =================================================================
class HMC_GNN(nn.Module):
    def __init__(self, initial_features, n_relations, hidden_dim=384, dropout_p=0.2):
        super().__init__()
        
        n_entities, input_dim = initial_features.shape
        
        # 1. 初始节点特征 (允许微调)
        self.node_features = nn.Embedding.from_pretrained(torch.from_numpy(initial_features).float(), freeze=False)
        
        # 2. GNN Encoder (RGCN)
        # Layer 1: Input Dim -> Hidden Dim
        self.conv1 = RGCNConv(input_dim, hidden_dim, n_relations * 2)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        
        # Layer 2: Hidden Dim -> Hidden Dim
        self.conv2 = RGCNConv(hidden_dim, hidden_dim, n_relations * 2)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        
        self.dropout = nn.Dropout(dropout_p)
        
        # 3. Decoder (RotatE)
        # RotatE 关系维度通常是实体维度的一半 (实部+虚部)
        self.rel_dim = hidden_dim // 2
        self.rel_emb = nn.Embedding(n_relations, self.rel_dim)
        
        emb_range = 6.0 / np.sqrt(self.rel_dim)
        nn.init.uniform_(self.rel_emb.weight.data, a=-emb_range, b=emb_range)

    def forward_encoder(self, x, edge_index, edge_type):
        """
        全图 GNN 前向传播，计算所有节点的 Embedding (z)
        """
        if x is None:
            x = self.node_features.weight
            
        x = self.conv1(x, edge_index, edge_type)
        x = self.bn1(x)
        x = F.elu(x)
        x = self.dropout(x)
        
        x = self.conv2(x, edge_index, edge_type)
        x = self.bn2(x)
        # KGE 常用归一化
        x = F.normalize(x, p=2, dim=1) 
        return x 

    def _scoring(self, h_emb, r_emb, t_emb):
        """
        RotatE 评分函数：返回 -distance (分数越高越好)
        """
        # 拆分实部和虚部
        h_re, h_im = h_emb.chunk(2, dim=-1)
        
        # 关系相位
        emb_range = 6.0 / np.sqrt(self.rel_dim)
        phase_r = r_emb / emb_range * np.pi
        r_re, r_im = torch.cos(phase_r), torch.sin(phase_r)

        # 评估模式 (t_emb 是候选集矩阵 [N, dim])
        if h_emb.dim() != t_emb.dim() or h_emb.shape[0] != t_emb.shape[0]:
            t_re, t_im = t_emb.chunk(2, dim=-1)
            
            # 广播 h 和 r: [B, 1, dim]
            h_re, h_im = h_re.unsqueeze(1), h_im.unsqueeze(1)
            r_re, r_im = r_re.unsqueeze(1), r_im.unsqueeze(1)
            
            # RotatE: h * r
            re_score = (h_re * r_re - h_im * r_im) - t_re 
            im_score = (h_re * r_im + h_im * r_re) - t_im
            
            # L2 距离
            dist = torch.sqrt(re_score**2 + im_score**2 + 1e-12).sum(dim=-1)
            return -dist 
            
        # 训练模式 (t_emb 是 batch [B, dim])
        else:
            t_re, t_im = t_emb.chunk(2, dim=-1)
            re_score = (h_re * r_re - h_im * r_im) - t_re
            im_score = (h_re * r_im + h_im * r_re) - t_im
            dist = torch.sqrt(re_score**2 + im_score**2 + 1e-12).sum(dim=-1)
            return -dist

    def forward_decoder(self, z, h_idx, r_idx, t_idx):
        """
        解码器前向传播
        """
        h_emb = z[h_idx]
        t_emb = z[t_idx]
        r_emb = self.rel_emb(r_idx)
        return self._scoring(h_emb, r_emb, t_emb)

# =================================================================
# 2. 评估函数 (带过滤和类型约束)
# =================================================================
@torch.no_grad()
def evaluate_gnn(model, z, loader, all_triples_map, relation2types, type2entity_ids_gpu, ix2rel, device):
    model.eval()
    ranks = []
    N = z.shape[0]

    for batch in tqdm(loader, desc="Evaluating"):
        h_idx, r_idx, t_idx = batch[0].to(device), batch[1].to(device), batch[2].to(device)
        batch_size = h_idx.shape[0]

        # 1. 计算所有候选分数
        h_emb = z[h_idx]
        r_emb = model.rel_emb(r_idx)
        # 对所有实体打分
        scores = model._scoring(h_emb, r_emb, z) 

        # 2. 应用约束和过滤
        for i in range(batch_size):
            h, r, t = h_idx[i].item(), r_idx[i].item(), t_idx[i].item()
            rel_name = ix2rel[r]

            # A. 类型约束 (Masking)
            if rel_name in relation2types:
                tail_type = relation2types[rel_name][1]
                if tail_type in type2entity_ids_gpu:
                    valid_candidates = type2entity_ids_gpu[tail_type]
                    
                    # 创建 Mask: 默认为 True (过滤掉)，只有 valid 为 False
                    type_mask = torch.ones(N, dtype=torch.bool, device=device)
                    type_mask[valid_candidates] = False
                    scores[i, type_mask] = -float('inf')

            # B. 已知事实过滤 (Filtering)
            filter_ids = all_triples_map.get((h, r), [])
            filter_mask = [fid for fid in filter_ids if fid != t]
            if filter_mask:
                scores[i, filter_mask] = -float('inf')

        # 3. 排序计算排名
        _, sorted_indices = torch.sort(scores, dim=1, descending=True) # 分数越高越好 (-distance)
        
        for i in range(batch_size):
            t = t_idx[i].item()
            rank_idx = (sorted_indices[i] == t).nonzero()
            if rank_idx.numel() > 0:
                rank = rank_idx.item() + 1
                ranks.append(rank)
            else:
                # 理论上不应发生，除非 t 被错误过滤
                ranks.append(N)

    ranks = np.array(ranks)
    return {
        "mrr": (1. / ranks).mean(), 
        "hits@1": (ranks <= 1).mean(), 
        "hits@10": (ranks <= 10).mean()
    }

# =================================================================
# 3. 主逻辑
# =================================================================
def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # --- 路径 ---
    SCRIPT_DIR = Path(__file__).parent.resolve()
    DATA_ROOT = SCRIPT_DIR / "dataset" / "NEWHERB"
    KGE_DIR = DATA_ROOT / "kge_data"
    FEATURE_DIR = SCRIPT_DIR / "output" / "NEWHERB"
    OUTPUT_DIR = SCRIPT_DIR / "output" / "hmc_gnn_model"
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

    # 2. 构建 PyG 图
    print("Building Graph...")
    src_list = [ent2ix[h] for h in df_train['from'] if h in ent2ix]
    dst_list = [ent2ix[t] for t in df_train['to'] if t in ent2ix]
    rels_list = [rel2ix[r] for r in df_train['rel'] if r in rel2ix]

    src = torch.tensor(src_list, dtype=torch.long)
    dst = torch.tensor(dst_list, dtype=torch.long)
    rels = torch.tensor(rels_list, dtype=torch.long)

    # 添加反向边 (Relation ID + n_relations)
    edge_index = torch.stack([
        torch.cat([src, dst]), 
        torch.cat([dst, src])
    ], dim=0).to(device)
    
    edge_type = torch.cat([rels, rels + n_relations]).to(device)

    # 3. 加载特征
    print("Loading features...")
    emb_path = FEATURE_DIR / "final_entity_embeddings_text_smiles.npy"
    final_embeddings = np.load(emb_path)
    assert final_embeddings.shape[0] == n_entities, "特征数量不匹配"
    initial_x = torch.from_numpy(final_embeddings).float().to(device)

    # 4. 加载约束
    print("Loading constraints...")
    entity2type = {p[0]: p[1] for p in (l.strip().split('\t') for l in open(KGE_DIR / "entity2type.txt", 'r', encoding='utf-8')) if len(p) == 2}
    relation2types = {p[0]: (p[1], p[2]) for p in (l.strip().split('\t') for l in open(KGE_DIR / "relation2types.txt", 'r', encoding='utf-8')) if len(p) == 3}

    type2entity_ids = defaultdict(list)
    for ent, etype in entity2type.items():
        if ent in ent2ix:
            type2entity_ids[etype].append(ent2ix[ent])
    
    # 创建 GPU 上的类型索引，加速评估
    type2entity_ids_gpu = {k: torch.tensor(v, dtype=torch.long, device=device) for k, v in type2entity_ids.items()}

    # 5. 过滤 Map
    all_triples_map = defaultdict(set)
    for df in [df_train, df_val, df_test]:
        for _, row in df.iterrows():
            if row['from'] in ent2ix and row['to'] in ent2ix:
                h, r, t = ent2ix[row['from']], rel2ix[row['rel']], ent2ix[row['to']]
                all_triples_map[(h, r)].add(t)
    all_triples_map = {k: list(v) for k, v in all_triples_map.items()}

    # 6. 初始化模型
    model = HMC_GNN(
        initial_features=final_embeddings,
        n_relations=n_relations,
        hidden_dim=384, 
        dropout_p=0.2
    ).to(device)

    # 使用 MarginRankingLoss (适用于 score 越高越好，target=1)
    criterion = MarginRankingLoss(margin=1.0)
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=5e-4)

    # 7. DataLoaders
    # 训练集 (只包含正样本 ID)
    train_ds = TensorDataset(
        torch.tensor(src_list, dtype=torch.long),
        torch.tensor(rels_list, dtype=torch.long),
        torch.tensor(dst_list, dtype=torch.long)
    )
    train_loader = DataLoader(train_ds, batch_size=1024, shuffle=True)
    
    # 验证集
    val_triples = [[ent2ix[row['from']], rel2ix[row['rel']], ent2ix[row['to']]] for _, row in df_val.iterrows()]
    val_loader = DataLoader(TensorDataset(
        torch.tensor([t[0] for t in val_triples]),
        torch.tensor([t[1] for t in val_triples]),
        torch.tensor([t[2] for t in val_triples])
    ), batch_size=128)
    
    # 测试集
    test_triples = [[ent2ix[row['from']], rel2ix[row['rel']], ent2ix[row['to']]] for _, row in df_test.iterrows()]
    test_loader = DataLoader(TensorDataset(
        torch.tensor([t[0] for t in test_triples]),
        torch.tensor([t[1] for t in test_triples]),
        torch.tensor([t[2] for t in test_triples])
    ), batch_size=128)

    # 8. 训练循环
    n_epochs = 300
    best_mrr = 0.0
    patience = 30
    no_improve = 0
    val_interval = 5

    print(f"\nStarting training HMC-GNN for up to {n_epochs} epochs...")

    for epoch in range(n_epochs):
        model.train()
        total_loss = 0.0

        # 1. 全图前向传播：获取该 epoch 的节点 embedding
        # z = model.forward_encoder(initial_x, edge_index, edge_type)

        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
            h_batch, r_batch, t_batch = batch[0].to(device), batch[1].to(device), batch[2].to(device)
            
            optimizer.zero_grad()

            z = model.forward_encoder(initial_x, edge_index, edge_type)
            
            # 正样本分数
            pos_scores = model.forward_decoder(z, h_batch, r_batch, t_batch)
            
            # 负采样 (向量化)
            # 随机生成负尾实体
            neg_t = torch.randint(0, n_entities, (h_batch.shape[0],), device=device)
            neg_scores = model.forward_decoder(z, h_batch, r_batch, neg_t)
            
            # Margin Loss: pos > neg => target = 1
            target = torch.ones_like(pos_scores, device=device)
            loss = criterion(pos_scores, neg_scores, target)
            
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1} Loss: {avg_loss:.6f}")

        # 验证
        if (epoch + 1) % val_interval == 0:
            print("Validating...")
            # 验证时也需要最新的 z
            model.eval()
            with torch.no_grad():
                z_eval = model.forward_encoder(initial_x, edge_index, edge_type)
                metrics = evaluate_gnn(model, z_eval, val_loader, all_triples_map, relation2types, type2entity_ids_gpu, ix2rel, device)
            print(f"Val Metrics: {metrics}")
            
            if metrics['mrr'] > best_mrr:
                best_mrr = metrics['mrr']
                no_improve = 0
                torch.save(model.state_dict(), OUTPUT_DIR / "best_gnn_model.pt")
                print(f"⭐ New Best Model Saved! MRR: {best_mrr:.4f}")
            else:
                no_improve += val_interval
                if no_improve >= patience:
                    print("Early stopping triggered.")
                    break

    # 9. 最终测试
    print("\n=== Final Testing ===")
    if (OUTPUT_DIR / "best_gnn_model.pt").exists():
        model.load_state_dict(torch.load(OUTPUT_DIR / "best_gnn_model.pt"))
    
    model.eval()
    with torch.no_grad():
        z_final = model.forward_encoder(initial_x, edge_index, edge_type)
        test_metrics = evaluate_gnn(model, z_final, test_loader, all_triples_map, relation2types, type2entity_ids_gpu, ix2rel, device)
    print(f"Test Results: {test_metrics}")

if __name__ == "__main__":
    main()