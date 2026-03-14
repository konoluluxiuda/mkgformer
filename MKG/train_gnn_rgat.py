# train_gnn.py (Final Version with RGCN)
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
import os
from tqdm import tqdm
from pathlib import Path
from collections import defaultdict

# ✅ 1. 导入 RGATConv
from torch_geometric.nn import RGATConv
from torch_geometric.loader import NeighborLoader
from torch_geometric.nn import RGCNConv

# =================================================================
# 1. 模型定义: RGCN for Link Prediction
# =================================================================
class RGCNLinkPredictor(nn.Module):
    def __init__(self, n_relations, emb_dim):
        super().__init__()
        
        hidden_dim = 200
        self.conv1 = RGCNConv(emb_dim, hidden_dim, n_relations)
        self.conv2 = RGCNConv(hidden_dim, emb_dim, n_relations)
        
        self.decoder_relation_embedding = nn.Embedding(n_relations, emb_dim)
        nn.init.xavier_uniform_(self.decoder_relation_embedding.weight)

    def forward_encoder(self, x, edge_index, edge_type):
        x = self.conv1(x, edge_index, edge_type)
        x = F.relu(x)
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv2(x, edge_index, edge_type)
        return x
    
    def forward_decoder(self, z, h_idx, t_idx, r_idx):
        h_emb = z[h_idx]
        t_emb = z[t_idx]
        r_emb = self.decoder_relation_embedding(r_idx)
        return torch.sum(h_emb * r_emb * t_emb, dim=-1)
# =================================================================
# 2. 为 GNN 定制的评估函数
# =================================================================
@torch.no_grad()
def evaluate_gnn(model, final_node_embeddings, dataset, all_triples_map, device, batch_size=128):
    model.eval()
    ranks = []
    
    # 将 EvalDataset 转换为标准的 TensorDataset 和 DataLoader
    h = torch.tensor(dataset.head_idx, dtype=torch.long)
    t = torch.tensor(dataset.tail_idx, dtype=torch.long)
    r = torch.tensor(dataset.relations, dtype=torch.long)
    eval_dataloader = DataLoader(TensorDataset(h, t, r), batch_size=batch_size)
    
    for h_idx, t_idx, r_idx in tqdm(eval_dataloader, desc="Evaluating"):
        h_idx, t_idx, r_idx = h_idx.to(device), t_idx.to(device), r_idx.to(device)
        
        # 使用 GNN 的最终输出 embedding (z) 来进行解码
        # z 在评估时就是 final_node_embeddings
        h_emb = final_node_embeddings[h_idx]
        
        # ✅ 核心修复：使用 () 而不是 [] 来调用 Embedding 层
        r_emb = model.decoder_relation_embedding(r_idx)
        
        # 计算与所有实体的分数: (h*r) @ all_t.T
        scores = (h_emb * r_emb) @ final_node_embeddings.transpose(0, 1)
        
        for i in range(h_idx.shape[0]):
            h_item, r_item, t_item = h_idx[i].item(), r_idx[i].item(), t_idx[i].item()
            filter_ids = all_triples_map.get((h_item, r_item), [])
            # 过滤掉其他正确答案
            filter_ids_tensor = torch.tensor([f_id for f_id in filter_ids if f_id != t_item], dtype=torch.long, device=device)
            if len(filter_ids_tensor) > 0:
                scores[i, filter_ids_tensor] = -float('inf')
        
        _, sorted_indices = torch.sort(scores, dim=1, descending=True)
        for i in range(h_idx.shape[0]):
            rank = (sorted_indices[i] == t_idx[i]).nonzero(as_tuple=True)[0].item() + 1
            ranks.append(rank)

    ranks = np.array(ranks)
    return {"mrr": (1. / ranks).mean(), "hits@1": (ranks <= 1).mean(), "hits@10": (ranks <= 10).mean(), "mean_rank": ranks.mean()}

# =================================================================
# 3. 主脚本逻辑
# =================================================================
def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    data_dir = Path("dataset/HERB")
    feature_dir = Path("output/HERB")
    output_dir = Path("output/compgcnrgat_model")
    output_dir.mkdir(exist_ok=True, parents=True)

    print("Loading data and creating graph...")
    df_train = pd.read_csv(data_dir / 'train.tsv', sep='\t', header=None, names=['from', 'rel', 'to'])
    df_val = pd.read_csv(data_dir / 'dev.tsv', sep='\t', header=None, names=['from', 'rel', 'to'])
    df_test = pd.read_csv(data_dir / 'test.tsv', sep='\t', header=None, names=['from', 'rel', 'to'])
    
    print("Building entity and relation mappings...")
    all_entities_list = sorted(list(set(df_train['from']) | set(df_train['to']) | set(df_val['from']) | set(df_val['to']) | set(df_test['from']) | set(df_test['to'])))
    all_relations_list = sorted(list(set(df_train['rel']) | set(df_val['rel']) | set(df_test['rel'])))
    ent2ix = {ent: i for i, ent in enumerate(all_entities_list)}
    rel2ix = {rel: i for i, rel in enumerate(all_relations_list)}
    ix2ent = {i: ent for ent, i in ent2ix.items()}
    n_entities = len(all_entities_list)
    n_relations = len(all_relations_list)
    print(f"Found {n_entities} unique entities and {n_relations} unique relations.")

    print("Loading and aligning pre-trained embeddings...")
    # (这部分假设你已经运行了 fuse_features.py, 并且有 final_entity_embeddings_text_smiles.npy)
    final_entity_embeddings_raw = np.load(feature_dir / "final_entity_embeddings_text_smiles.npy")
    with open(data_dir / "entities.txt", 'r', encoding='utf-8') as f:
        original_entity_list = [line.strip() for line in f]
    original_entity_map = {name: i for i, name in enumerate(original_entity_list)}
    
    initial_node_features = np.zeros((n_entities, final_entity_embeddings_raw.shape[1]))
    for i, entity_name in enumerate(all_entities_list):
        if entity_name in original_entity_map:
            initial_node_features[i] = final_entity_embeddings_raw[original_entity_map[entity_name]]
    
    # --- 准备图数据和训练/评估数据 ---
    print("Preparing graph and training/evaluation data...")
    source_nodes = torch.tensor([ent2ix[h] for h in df_train['from']], dtype=torch.long)
    target_nodes = torch.tensor([ent2ix[t] for t in df_train['to']], dtype=torch.long)
    relations = torch.tensor([rel2ix[r] for r in df_train['rel']], dtype=torch.long)
    
    # 完整图的边
    edge_index = torch.stack([torch.cat([source_nodes, target_nodes]), torch.cat([target_nodes, source_nodes])], dim=0)
    inverse_relations = relations + n_relations
    edge_type = torch.cat([relations, inverse_relations])
    
    # 初始节点特征
    initial_x = torch.from_numpy(initial_node_features).float()
    
    # 将图数据移动到 GPU
    edge_index = edge_index.to(device)
    edge_type = edge_type.to(device)
    initial_x = initial_x.to(device)

    # 创建 DataLoader 用于训练
    train_dataset = torch.stack([source_nodes, relations, target_nodes], dim=1)
    train_dataloader = DataLoader(train_dataset, batch_size=1024, shuffle=True)

    # --- 初始化 RGCN 模型 ---
    model = RGCNLinkPredictor(
        n_relations=n_relations * 2,
        emb_dim=initial_node_features.shape[1]
    ).to(device)

    initial_x = torch.from_numpy(initial_node_features).float().to(device)
    edge_index = edge_index.to(device)
    edge_type = edge_type.to(device)
    
    criterion = nn.BCEWithLogitsLoss()
    # ✅ 2. 优化超参数
    optimizer = optim.Adam(model.parameters(), lr=0.005, weight_decay=5e-4)
    
    train_triples = torch.stack([source_nodes, relations, target_nodes], dim=1)
    train_dataset = TensorDataset(train_triples)
    train_dataloader = DataLoader(train_dataset, batch_size=1024, shuffle=True)

    # --- 训练循环 (修正为全图范式) ---
    n_epochs = 200
    print(f"\nStarting training for {n_epochs} epochs with RGCN...")
    for epoch in range(n_epochs):
        model.train()
        running_loss = 0.0
        
        # GNN Encoder: 在每个 epoch 开始时，计算一次完整的 z
        z = model.forward_encoder(initial_x, edge_index, edge_type)
        
        for (batch_triples,) in tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{n_epochs}"):
            # ✅ 核心修复：将 batch 数据移动到 GPU
            batch_triples = batch_triples.to(device)
            
            optimizer.zero_grad()
            
            pos_h, pos_r, pos_t = batch_triples[:, 0], batch_triples[:, 1], batch_triples[:, 2]
            
            # 采样负样本
            neg_t = torch.randint(0, n_entities, (len(pos_h),), device=device)
            
            # 在 GNN 的输出 z 上进行解码
            pos_scores = model.forward_decoder(z, pos_h, pos_t, pos_r)
            neg_scores = model.forward_decoder(z, pos_h, neg_t, pos_r)
            
            scores = torch.cat([pos_scores, neg_scores])
            targets = torch.cat([torch.ones_like(pos_scores), torch.zeros_like(neg_scores)])
            
            loss = criterion(scores, targets)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            
        print(f'Epoch {epoch+1}/{n_epochs}, Loss: {running_loss / len(train_dataloader):.6f}')
    # --- 评估 ---
    print("\nTraining complete. Evaluating model...")
    # 在评估前，计算一次最终的、完整的节点 embedding
    final_node_embeddings = model.forward_encoder(initial_x, edge_index, edge_type).cpu() # 移到 CPU
    
    # 为了评估，需要将 pandas dataframe 转换为 KGDataset-like object for our evaluator
    class EvalDataset:
        def __init__(self, df, ent2ix, rel2ix):
            self.head_idx = [ent2ix[h] for h in df['from']]
            self.tail_idx = [ent2ix[t] for t in df['to']]
            self.relations = [rel2ix[r] for r in df['rel']]
        def __len__(self): return len(self.head_idx)

    val_dataset = EvalDataset(df_val, ent2ix, rel2ix)
    test_dataset = EvalDataset(df_test, ent2ix, rel2ix)
    
    all_triples_map = defaultdict(list)
    for df in [df_train, df_val, df_test]:
        for _, row in df.iterrows():
            if row['from'] in ent2ix and row['rel'] in rel2ix and row['to'] in ent2ix:
                h, r, t = ent2ix[row['from']], rel2ix[row['rel']], ent2ix[row['to']]
                all_triples_map[(h, r)].append(t)

    print("\nEvaluating on validation set...")
    val_metrics = evaluate_gnn(model, final_node_embeddings, val_dataset, all_triples_map, device)
    print(f"Validation Results: {val_metrics}")
    
    print("\nEvaluating on test set...")
    test_metrics = evaluate_gnn(model, final_node_embeddings, test_dataset, all_triples_map, device)
    print(f"Test Results: {test_metrics}")

if __name__ == "__main__":
    main()