# train_kge_conve.py (Implementation with ConvE scoring)
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
# ✅ 1. 从 torch.nn 导入 BCEWithLogitsLoss
from torch.nn import BCEWithLogitsLoss
import numpy as np
import pandas as pd
import os
from tqdm import tqdm
from collections import defaultdict
from pathlib import Path

# =================================================================
# 1. 自定义数据集和负采样
# =================================================================
class KGDataset(Dataset):
    def __init__(self, df, ent2ix, rel2ix):
        self.head_idx = torch.tensor([ent2ix[h] for h in df['from']], dtype=torch.long)
        self.tail_idx = torch.tensor([ent2ix[t] for t in df['to']], dtype=torch.long)
        self.relations = torch.tensor([rel2ix[r] for r in df['rel']], dtype=torch.long)
    
    def __len__(self):
        return len(self.head_idx)
        
    def __getitem__(self, idx):
        return self.head_idx[idx], self.tail_idx[idx], self.relations[idx]

def uniform_negative_sampler(h_batch, t_batch, num_entities, device):
    batch_size = h_batch.shape[0]
    corrupt_head_mask = torch.rand(batch_size, device=device) > 0.5
    neg_indices = torch.randint(0, num_entities, (batch_size,), device=device)
    n_h = torch.where(corrupt_head_mask, neg_indices, h_batch)
    n_t = torch.where(~corrupt_head_mask, neg_indices, t_batch)
    return n_h, n_t

# =================================================================
# 2. 模型定义: ModalRotatE
# =================================================================
class ModalConvE(nn.Module):
    def __init__(self, text_embeddings, smiles_embeddings, component_indices_map, n_relations, emb_dim):
        super().__init__()
        self.text_ent_emb = nn.Embedding.from_pretrained(torch.from_numpy(text_embeddings).float(), freeze=False)
        self.smiles_ent_emb = nn.Embedding.from_pretrained(torch.from_numpy(smiles_embeddings).float(), freeze=False)
        self.rel_emb = nn.Embedding(n_relations, emb_dim)
        nn.init.xavier_uniform_(self.rel_emb.weight.data)
        self.component_indices_map = component_indices_map
        # ✅ 补全 gate_layer 定义
        self.gate_layer = nn.Sequential(
            nn.Linear(emb_dim * 2, emb_dim // 4), nn.ReLU(),
            nn.Linear(emb_dim // 4, 1), nn.Sigmoid()
        )

        self.emb_dim = emb_dim
        self.emb_h = 24
        self.emb_w = 32
        assert self.emb_h * self.emb_w == self.emb_dim
        
        self.inp_drop = nn.Dropout(0.2)
        self.hidden_drop = nn.Dropout(0.3)
        self.feature_map_drop = nn.Dropout(0.2)
        
        self.conv1 = nn.Conv2d(1, 32, (3, 3), 1, 0)
        self.bn0 = nn.BatchNorm2d(1)
        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm1d(self.emb_dim)

        conv_out_h = 2 * self.emb_h - 2
        conv_out_w = self.emb_w - 2
        flat_size = conv_out_h * conv_out_w * 32
        self.fc = nn.Linear(flat_size, self.emb_dim)

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
    
    def _scoring(self, h_emb, r_emb, all_t_emb=None):
        # 1. Reshape h 和 r
        h_emb_reshaped = h_emb.view(-1, 1, self.emb_h, self.emb_w)
        r_emb_reshaped = r_emb.view(-1, 1, self.emb_h, self.emb_w)
        
        # 2. 上下拼接成一个 "图像" [batch_size, 1, emb_h*2, emb_w]
        stk_inp = torch.cat([h_emb_reshaped, r_emb_reshaped], 2)
        x = self.bn0(stk_inp)
        x = self.inp_drop(x)

        # 3. 卷积
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.feature_map_drop(x)
        
        # 4. 展平和投影
        x = x.view(x.shape[0], -1) # Flatten
        x = self.fc(x)
        x = self.hidden_drop(x)
        x = self.bn2(x)
        x = F.relu(x)
        
        # 5. 匹配
        # 训练时，all_t_emb 是 None，我们返回 x 供 forward 计算
        if all_t_emb is None:
            return x
        # 评估时，计算与所有候选 t 的点积
        else:
            return x @ all_t_emb.transpose(0, 1)

    def forward(self, h_idx, r_idx):
        # forward 方法现在只负责计算 (h,r) 的组合特征
        h_emb = self.get_entity_embeddings(h_idx)
        r_emb = self.rel_emb(r_idx)
        
        h_emb_reshaped = h_emb.view(-1, 1, self.emb_h, self.emb_w)
        r_emb_reshaped = r_emb.view(-1, 1, self.emb_h, self.emb_w)
        
        stk_inp = torch.cat([h_emb_reshaped, r_emb_reshaped], 2)
        x = self.bn0(stk_inp)
        x = self.inp_drop(x)
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.feature_map_drop(x)
        x = x.view(x.shape[0], -1)
        x = self.fc(x)
        x = self.hidden_drop(x)
        x = self.bn2(x)
        x = F.relu(x)
        return x


# =================================================================
# 3. 自定义评估函数
# =================================================================
@torch.no_grad()
def evaluate_conve(model, dataset, all_triples_map, device, batch_size=128):
    model.eval()
    ranks = []
    all_entity_ids = torch.arange(model.text_ent_emb.num_embeddings).to(device)
    all_entity_embs_fused = model.get_entity_embeddings(all_entity_ids)

    for h_idx, t_idx, r_idx in tqdm(DataLoader(dataset, batch_size=batch_size), desc="Evaluating"):
        h_idx, t_idx, r_idx = h_idx.to(device), t_idx.to(device), r_idx.to(device)
        
        # 计算 (h,r) 特征
        hr_feat = model(h_idx, r_idx)
        
        # 计算与所有实体的分数
        scores = hr_feat @ all_entity_embs_fused.transpose(0, 1) # -> [batch_size, num_entities]
        
        for i in range(h_idx.shape[0]):
            h, r, t = h_idx[i].item(), r_idx[i].item(), t_idx[i].item()
            filter_ids = all_triples_map.get((h, r), [])
            # 分数越高越好，所以把其他正确答案的分数设为极低值
            scores[i, [f_id for f_id in filter_ids if f_id != t]] = -float('inf')
        
        _, sorted_indices = torch.sort(scores, dim=1, descending=True)
        for i in range(h_idx.shape[0]):
            rank = (sorted_indices[i] == t_idx[i]).nonzero(as_tuple=True)[0].item() + 1
            ranks.append(rank)

    model.train()
    ranks = np.array(ranks)
    return {"mrr": (1. / ranks).mean(), "hits@1": (ranks <= 1).mean(), "hits@10": (ranks <= 10).mean()}

# =================================================================
# 4. 主脚本逻辑
# =================================================================
def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # --- 路径配置 ---
    data_dir = Path("dataset/HERB")
    feature_dir = Path("output/HERB")
    output_dir = Path("output/rotate_manual_model")
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
    model = ModalConvE(
        emb_dim=text_embeddings.shape[1],
        n_relations=n_relations,
        text_embeddings=text_embeddings,
        smiles_embeddings=smiles_embeddings,
        component_indices_map=component_indices_map,
    )
    
    # ConvE 通常与 BCEWithLogitsLoss 搭配
    criterion = BCEWithLogitsLoss()
    model.to(device)
    criterion.to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=0.0) # ConvE 通常需要更大的 lr 和更小的 wd
    
    # --- 训练循环 ---
    n_epochs = 500

    print(f"\nStarting training for {n_epochs} epochs with ModalConvE...")


    for epoch in range(n_epochs):
        running_loss = 0.0
        model.train()
        for h, t, r in tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{n_epochs}"):
            h, t, r = h.to(device), t.to(device), r.to(device)
            optimizer.zero_grad()
            
            # 计算 (h,r) 组合特征
            hr_feat = model(h_idx=h, r_idx=r)
            
            # 获取正样本 t 的 embedding
            t_emb = model.get_entity_embeddings(t)
            
            # 计算正样本分数
            pos_scores = torch.sum(hr_feat * t_emb, dim=1)
            
            # 采样负样本并计算分数
            n_h, n_t = uniform_negative_sampler(h, t, n_entities, device)
            n_t_emb = model.get_entity_embeddings(n_t)
            neg_scores = torch.sum(hr_feat * n_t_emb, dim=1)
            
            # 使用 BCEWithLogitsLoss
            scores = torch.cat([pos_scores, neg_scores])
            targets = torch.cat([torch.ones_like(pos_scores), torch.zeros_like(neg_scores)])
            
            loss = criterion(scores, targets)
            loss.backward()
            optimizer.step()
            
            # 移除归一化，ConvE 不需要
            running_loss += loss.item()
        
        epoch_loss = running_loss / len(train_dataloader)
        print(f'Epoch {epoch+1}/{n_epochs}, Loss: {epoch_loss:.6f}')
    
    # --- 评估 ---
    print("\nEvaluating on validation set...")
    val_metrics = evaluate_conve(model, val_dataset, all_triples_map, device)
    print(f"Validation Results: {val_metrics}")
    
    print("\nEvaluating on test set...")
    test_metrics = evaluate_conve(model, test_dataset, all_triples_map, device)
    print(f"Test Results: {test_metrics}")

if __name__ == "__main__":
    main()