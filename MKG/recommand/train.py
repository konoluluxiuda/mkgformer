from pickle import FALSE
from re import T
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
import sys
import numpy as np

# 导入本地模块
from config import Config
from dataset import GraphDataManager, HerbRecDataset
from model import HMC_GNN_SSL
from utils import set_seed, Evaluator

def main():
    # =========================================================================
    # 实验配置开关 (Experimental Switches)
    # 修改这里的 True/False 来组合不同的策略
    # =========================================================================

    # 1. 图结构选择
    # True: 使用包含 "文本相似度" 边的图 (preprocess_semantic_graph.py 生成)
    # False: 使用包含 "协作/共现" 边的图 (preprocess_kge.py 生成)
    USE_TFIDF_GRAPH = False
    USE_SEMANTIC_GRAPH = False


    # 2. 特征注入选择 (多模态融合)
    # True: 注入 "性味、归经" Multi-hot 向量 (这是之前 SOTA 的核心)
    USE_BASE_ATTR = False    

    # 开启跨模态对比学习
    USE_CROSS_MODAL = False
    # 权重系数
    CROSS_MODAL_WEIGHT = 0.2   

    # True: 
    USE_CHEM_DENSE = False   

    # =========================================================================

    set_seed(Config.seed)
    print(f"\n{'='*40}")
    print(f"Starting Training on device: {Config.device}")
    print(f"Strategy Config:")
    print(f"  [Graph] Semantic Graph: {USE_SEMANTIC_GRAPH}")
    print(f"  [Feat]  Base Attributes (Property/Meridian): {USE_BASE_ATTR}")
    print(f"  [Feat]  Deep Chemical (BERT/SMILES): {USE_CHEM_DENSE}")
    print(f"{'='*40}\n")

    # --- 1. 动态路径调整 ---
    if USE_TFIDF_GRAPH:
        print(">>> [Experiment] Loading TF-IDF GRAPH (Anti-Hub Strategy)...")
        # 指向刚才生成的数据目录
        Config.REC_DATA_DIR = os.path.join(Config.DATA_ROOT, 'tfidf_graph_data')
    elif USE_SEMANTIC_GRAPH:
        print(">>> Loading SEMANTIC GRAPH data...")
        Config.REC_DATA_DIR = os.path.join(Config.DATA_ROOT, 'semantic_data')
    else:
        print(">>> Loading ORIGINAL COLLABORATIVE GRAPH data...")
        Config.REC_DATA_DIR = os.path.join(Config.DATA_ROOT, 'recommendation_data')
        
    # --- 2. 加载图结构数据 ---
    data_manager = GraphDataManager()
    try:
        # load_data 会读取 REC_DATA_DIR 下的 edge_index, edge_type, rec_data.pt
        edge_index, edge_type, train_dict, test_dict = data_manager.load_data()
    except FileNotFoundError as e:
        print(f"❌ Error loading graph data: {e}")
        print("Please run 'preprocess_kge.py' or 'preprocess_semantic_graph.py' first.")
        return

        # 3. [新增] 加载深层化学特征 (BERT/ChemBERTa)
    chem_matrix = None
    if USE_CROSS_MODAL:
        chem_path = os.path.join(Config.DATA_ROOT, 'recommendation_data', 'node_chem_dense.pt')
        if os.path.exists(chem_path):
            chem_matrix = torch.load(chem_path).to(Config.device)
            print(f"✅ Loaded Cross-Modal Chem Matrix: {chem_matrix.shape}")
        else:
            print("⚠️ Chem Matrix not found, Cross-Modal SSL will be disabled.")

    # --- 3. 准备特征矩阵 (Attribute Injection) ---
    attr_tensors = []

    # A. 加载基础属性 (Property/Meridian)
    if USE_BASE_ATTR:
        base_attr = data_manager.load_attributes() # 读取 node_attributes.pt
        if base_attr is not None:
            print(f"✅ Loaded Base Attributes: {base_attr.shape}")
            attr_tensors.append(base_attr)
        else:
            print("⚠️ Warning: Base attributes enabled but file not found.")

    # B. 加载深度化学特征 (Dense BERT/SMILES)
    if USE_CHEM_DENSE:
        chem_path = os.path.join(Config.DATA_ROOT, 'recommendation_data', 'node_chem_multihot.pt')
        if os.path.exists(chem_path):
            chem_attr = torch.load(chem_path)
            print(f"✅ Loaded Deep Chemical Features: {chem_attr.shape}")
            attr_tensors.append(chem_attr)
        else:
            print(f"⚠️ Warning: Deep Chemical file not found at {chem_path}")
            print("   Please run 'build_deep_chem.py' first.")

    # C. 特征拼接
    if attr_tensors:
        # 在特征维度 (dim=1) 进行拼接
        final_attr_matrix = torch.cat(attr_tensors, dim=1).to(Config.device)
        print(f"🔹 Final Attribute Matrix Shape: {final_attr_matrix.shape}")
    else:
        final_attr_matrix = None
        print("🔹 No external attributes used (Pure Structure Learning).")

    # --- 4. 准备 DataLoader ---
    train_dataset = HerbRecDataset(train_dict, data_manager.herb_indices)
    train_loader = DataLoader(train_dataset, batch_size=Config.batch_size, shuffle=True)

    # 将图移至 GPU
    edge_index = edge_index.to(Config.device)
    edge_type = edge_type.to(Config.device)

    # --- 5. 初始化模型 ---
    # 注意: HMC_GNN_SSL 会自动检测 attr_matrix 的维度并初始化 Linear 层
    model = HMC_GNN_SSL(
        num_nodes=data_manager.num_nodes,
        num_relations=data_manager.num_relations,
        pretrained_features=None,    # 始终为 None (我们要保持 Random ID Embedding)
        attr_matrix=final_attr_matrix, # 传入拼接好的属性
        chem_matrix=chem_matrix  # <--- 传入化学矩阵
    ).to(Config.device)

    # 优化器
    # 由于我们使用的是 Attribute Injection (拼接策略)，所有参数(GCN, Linear, Embedding)
    # 都可以使用统一的学习率，不需要分层，因为 Linear 层会自动适配特征分布。
    optimizer = optim.Adam(model.parameters(), lr=Config.lr, weight_decay=Config.weight_decay)

    evaluator = Evaluator()

    # --- 6. 训练循环 (带早停) ---
    best_f1 = 0.0
    no_improve_cnt = 0

    print(f"\nStart Training... (Max Epochs: {Config.epochs}, Patience: {Config.patience})")

    for epoch in range(Config.epochs):
        model.train()
        total_loss = 0.0
        
        # 进度条
        with tqdm(train_loader, desc=f"Epoch {epoch+1}/{Config.epochs}", unit="batch", leave=False) as tepoch:
            for batch in tepoch:
                u, pos, neg = batch
                u, pos, neg = u.to(Config.device), pos.to(Config.device), neg.to(Config.device)
                
                optimizer.zero_grad()
                
                # --- Forward (SSL Dual Views) ---
                # perturbed=True 会触发 Edge Dropout，生成两个略微不同的视图
                x_view1 = model.forward_encoder(edge_index, edge_type, perturbed=True)
                x_view2 = model.forward_encoder(edge_index, edge_type, perturbed=True)
                
                # --- Task 1: Recommendation Loss (BPR) ---
                # 使用 View 1 的特征进行推荐
                # A. 推荐 BPR Loss
                u_emb, pos_emb, neg_emb = x_view1[u], x_view1[pos], x_view1[neg]
                pos_scores = (u_emb * pos_emb).sum(dim=1)
                neg_scores = (u_emb * neg_emb).sum(dim=1)
                bpr_loss = -torch.mean(torch.nn.functional.logsigmoid(pos_scores - neg_scores))
                
                # B. 图内扰动对比 Loss (Graph SSL)
                unique_nodes = torch.unique(torch.cat([u, pos, neg]))
                graph_ssl_loss = model.calc_ssl_loss(x_view1, x_view2, unique_nodes)
                
                # C. [新增] 跨模态对比 Loss (Cross-Modal SSL)
                cm_ssl_loss = torch.tensor(0.0, device=Config.device)
                if USE_CROSS_MODAL:
                    # 仅对草药节点计算跨模态损失
                    unique_herbs = torch.unique(torch.cat([pos, neg]))
                    # 我们用无扰动状态的特征或者 View1 的特征去逼近 Chemical
                    cm_ssl_loss = model.calc_cross_modal_loss(x_view1, unique_herbs)
                
                # 总 Loss
                loss = bpr_loss + Config.ssl_reg * graph_ssl_loss + CROSS_MODAL_WEIGHT * cm_ssl_loss
                
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                tepoch.set_postfix(loss=loss.item())
            
        avg_loss = total_loss / len(train_loader)
        
        # --- 7. 评估与早停 ---
        if (epoch + 1) % Config.eval_interval == 0:
            print(f"Epoch {epoch+1} | Loss: {avg_loss:.4f}")
            
            # 评估
            results = evaluator.evaluate(
                model, 
                test_dict, 
                data_manager.herb_indices, 
                edge_index, 
                edge_type
            )
            
            # 格式化输出
            res_str = " | ".join([f"{k}: {v:.4f}" for k, v in results.items() if 'F1' in k])
            print(f"   >> Test Metrics: {res_str}")
            
            cur_f1 = results['F1@10']
            
            # 保存最佳模型
            if cur_f1 > best_f1:
                best_f1 = cur_f1
                no_improve_cnt = 0
                save_path = os.path.join(Config.MODEL_SAVE_PATH, 'best_model.pt')
                torch.save(model.state_dict(), save_path)
                print(f"   >> ⭐ New Best Model! F1@10: {best_f1:.4f}")
            else:
                no_improve_cnt += 1
                print(f"   >> No improvement. Counter: {no_improve_cnt}/{Config.patience}")
                
                if no_improve_cnt >= Config.patience:
                    print(f"\n[Early Stopping] Triggered after {no_improve_cnt*Config.eval_interval} epochs without improvement.")
                    print(f"Training Finished. Best F1@10: {best_f1:.4f}")
                    break

if __name__ == "__main__":
    main()