import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
import numpy as np

# 导入配置
from config import Config
from dataset import HerbRecDataset
# 导入拆分数据加载器
from dataset_split import SplitGraphDataManager
# 导入拆分模型
from model_split import MultiView_GNN
# 导入新的评估器
from utils import set_seed, SplitEvaluator 

def main():
    set_seed(Config.seed)
    print(f"\n{'='*40}")
    print(f"Training Physically Split Multi-View Model (S-H, S-S, H-H)")
    print(f"Device: {Config.device}")
    print(f"{'='*40}\n")
    
    # 1. 加载 Split 数据 (基于 paper_graph_data 的 Top-K 结构)
    manager = SplitGraphDataManager()
    data_pack = manager.load_split_data()
    
    # 提取图并移至 GPU
    graphs = {
        'sh': data_pack['sh_graph'].to(Config.device),
        'ss': data_pack['ss_graph'].to(Config.device),
        'hh': data_pack['hh_graph'].to(Config.device)
    }
    
    # 加载属性 (复用 SOTA 策略: 显式属性注入)
    # 注意: model_split.py 内部会将属性仅注入到 H-H 分支
    attr_matrix = manager.load_attributes()
    if attr_matrix is not None:
        attr_matrix = attr_matrix.to(Config.device)
        print(f"✅ Attribute Matrix Loaded: {attr_matrix.shape}")
    else:
        print("⚠️ Attribute Matrix NOT Found.")

    # 2. Dataset
    train_dataset = HerbRecDataset(data_pack['train_dict'], data_pack['herb_indices'])
    train_loader = DataLoader(train_dataset, batch_size=Config.batch_size, shuffle=True)
    
    # 3. 初始化多视图模型
    model = MultiView_GNN(
        num_nodes=data_pack['num_nodes'], 
        attr_matrix=attr_matrix
    ).to(Config.device)
    
    optimizer = optim.Adam(model.parameters(), lr=Config.lr, weight_decay=Config.weight_decay)
    
    # 使用专门的拆分评估器
    evaluator = SplitEvaluator(k_list=[5, 10, 20])
    
    # 4. 训练循环
    best_f1 = 0.0
    no_improve = 0
    
    print(f"\nStart Training... (Max Epochs: {Config.epochs}, Patience: {Config.patience})")
    
    for epoch in range(Config.epochs):
        model.train()
        total_loss = 0.0
        
        with tqdm(train_loader, desc=f"Epoch {epoch+1}", unit="batch", leave=False) as tepoch:
            for batch in tepoch:
                u, pos, neg = batch
                u, pos, neg = u.to(Config.device), pos.to(Config.device), neg.to(Config.device)
                
                optimizer.zero_grad()
                
                # Forward (传入 graphs 字典)
                # View 1: 正常图 (perturbed=True 开启 Edge Dropout)
                x1 = model.forward_encoder(graphs, perturbed=True)
                # View 2: 扰动图
                x2 = model.forward_encoder(graphs, perturbed=True)
                
                # BPR Loss
                u_emb, pos_emb, neg_emb = x1[u], x1[pos], x1[neg]
                pos_scores = (u_emb * pos_emb).sum(dim=1)
                neg_scores = (u_emb * neg_emb).sum(dim=1)
                bpr_loss = -torch.mean(torch.nn.functional.logsigmoid(pos_scores - neg_scores))
                
                # SSL Loss
                nodes = torch.unique(torch.cat([u, pos, neg]))
                ssl_loss = model.calc_ssl_loss(x1, x2, nodes)
                
                loss = bpr_loss + Config.ssl_reg * ssl_loss
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                tepoch.set_postfix(loss=loss.item())
                
        # 评估
        if (epoch + 1) % Config.eval_interval == 0:
            avg_loss = total_loss / len(train_loader)
            print(f"Epoch {epoch+1} | Loss: {avg_loss:.4f}")
            
            # 使用完整的评估器
            results = evaluator.evaluate(
                model, 
                graphs, 
                data_pack['test_dict'], 
                data_pack['herb_indices'], 
                Config.device
            )
            
            # 格式化输出 (重点关注 F1 和 Recall)
            f1_str = " | ".join([f"F1@{k}: {results[f'F1@{k}']:.4f}" for k in [5, 10, 20]])
            rec_str = " | ".join([f"R@{k}: {results[f'Recall@{k}']:.4f}" for k in [5, 10, 20]])
            
            print(f"   >> {f1_str}")
            print(f"   >> {rec_str}")
            
            cur_f1 = results['F1@10']
            
            if cur_f1 > best_f1:
                best_f1 = cur_f1
                no_improve = 0
                torch.save(model.state_dict(), os.path.join(Config.MODEL_SAVE_PATH, 'best_split_model.pt'))
                print(f"   >> ⭐ New Best Model! F1@10: {best_f1:.4f}")
            else:
                no_improve += 1
                print(f"   >> No improvement. Counter: {no_improve}/{Config.patience}")
                
                if no_improve >= Config.patience:
                    print(f"\n[Early Stopping] Training Finished. Best F1@10: {best_f1:.4f}")
                    break

if __name__ == "__main__":
    main()