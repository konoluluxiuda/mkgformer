import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
import sys
import numpy as np

from config import Config
from dataset import HerbRecDataset
from dataset_split import SplitGraphDataManager
from model_split import MultiView_GNN
from utils import set_seed, SplitEvaluator

def main():
    # =========================================================================
    # 实验配置开关 (严格对齐主实验 train.py)
    # =========================================================================
    USE_TFIDF_GRAPH = False
    USE_SEMANTIC_GRAPH = False

    USE_BASE_ATTR = True    
    FUSION_MODE = 'gated'

    USE_CROSS_MODAL = True
    CROSS_MODAL_WEIGHT = 0.2   

    USE_PROP_CHEM_ALIGN = True
    PROP_CHEM_WEIGHT = 0.1

    USE_CHEM_DENSE = True
    USE_CHEM_FINGERPRINT = True
    USE_DISEASE_TEXT = True

    set_seed(Config.seed)
    print(f"\n{'='*40}")
    print(f"Starting Training on device: {Config.device}")
    print(f"Strategy Config: [Ablation: SPLIT MULTI-VIEW GRAPH]")
    print(f"  [Graph] Semantic Graph: {USE_SEMANTIC_GRAPH}")
    print(f"  [Fuse] Fusion Mode: {FUSION_MODE}")
    print(f"  [Feat] Base Attributes: {USE_BASE_ATTR}")
    print(f"  [Feat] Deep Chemical: {USE_CHEM_DENSE}")
    print(f"  [SSL]  Cross Modal: {USE_CROSS_MODAL}")
    print(f"  [SSL]  Property-Chem Align: {USE_PROP_CHEM_ALIGN}")
    print(f"{'='*40}\n")

    # --- 1. 动态路径调整 (同 train.py) ---
    if USE_TFIDF_GRAPH:
        Config.REC_DATA_DIR = os.path.join(Config.DATA_ROOT, 'tfidf_graph_data')
    elif USE_SEMANTIC_GRAPH:
        Config.REC_DATA_DIR = os.path.join(Config.DATA_ROOT, 'semantic_data')
    else:
        Config.REC_DATA_DIR = os.path.join(Config.DATA_ROOT, 'recommendation_data')
        
    manager = SplitGraphDataManager()
    data_pack = manager.load_split_data()
    
    graphs = {
        'sh': data_pack['sh_graph'].to(Config.device),
        'ss': data_pack['ss_graph'].to(Config.device),
        'hh': data_pack['hh_graph'].to(Config.device)
    }

    # --- 2. 加载完全相同的多模态特征矩阵 ---
    chem_matrix = None
    if USE_CROSS_MODAL:
        chem_path = os.path.join(Config.DATA_ROOT, 'recommendation_data', 'node_chem_dense.pt')
        if os.path.exists(chem_path):
            chem_matrix = torch.load(chem_path)
            if USE_CHEM_FINGERPRINT:
                fp_path_pt = os.path.join(Config.DATA_ROOT, 'recommendation_data', 'node_chem_fingerprint.pt')
                fp_path_npy = os.path.join(Config.DATA_ROOT, 'recommendation_data', 'node_chem_fingerprint.npy')
                fp_feat = None
                if os.path.exists(fp_path_pt):
                    fp_feat = torch.load(fp_path_pt)
                elif os.path.exists(fp_path_npy):
                    fp_feat = torch.from_numpy(np.load(fp_path_npy)).float()

                if fp_feat is not None:
                    chem_matrix = torch.cat([chem_matrix, fp_feat], dim=1)
            chem_matrix = chem_matrix.to(Config.device)
            print(f"✅ Loaded Cross-Modal Chem Matrix: {chem_matrix.shape}")

    attr_tensors = []
    if USE_BASE_ATTR:
        base_attr = torch.load(os.path.join(Config.DATA_ROOT, 'recommendation_data', 'node_attributes.pt'))
        if base_attr is not None:
            attr_tensors.append(base_attr)

    if USE_CHEM_DENSE:
        chem_attr = torch.load(os.path.join(Config.DATA_ROOT, 'recommendation_data', 'node_chem_multihot.pt'))
        if chem_attr is not None:
            attr_tensors.append(chem_attr)

    if attr_tensors:
        final_attr_matrix = torch.cat(attr_tensors, dim=1).to(Config.device)
    else:
        final_attr_matrix = None

    disease_matrix = None
    if USE_DISEASE_TEXT:
        disease_path = os.path.join(Config.DATA_ROOT, 'recommendation_data', 'node_disease_text.pt')
        if os.path.exists(disease_path):
            disease_matrix = torch.load(disease_path).to(Config.device)

    # --- 3. 初始化 DataLoader ------------------
    train_dataset = HerbRecDataset(data_pack['train_dict'], data_pack['herb_indices'])
    train_loader = DataLoader(train_dataset, batch_size=Config.batch_size, shuffle=True)

    # --- 4. 初始化模型 (参数与 train.py 一致) ---
    model = MultiView_GNN(
        num_nodes=data_pack['num_nodes'],
        num_relations=12,
        pretrained_features=None,
        attr_matrix=final_attr_matrix,
        chem_matrix=chem_matrix,
        disease_matrix=disease_matrix,
        fusion_mode=FUSION_MODE
    ).to(Config.device)

    optimizer = optim.Adam(model.parameters(), lr=Config.lr, weight_decay=Config.weight_decay)
    
    # 严格对齐 eval interval 和 metrics
    evaluator = SplitEvaluator(k_list=Config.top_k)
    save_path = os.path.join(Config.MODEL_SAVE_PATH, 'best_split_model.pt')
    
    best_f1 = 0.0
    no_improve_cnt = 0
    
    print(f"\nStart Training... (Max Epochs: {Config.epochs}, Patience: {Config.patience})")
    
    for epoch in range(Config.epochs):
        model.train()
        total_loss = 0.0
        
        with tqdm(train_loader, desc=f"Epoch {epoch+1}", unit="batch", leave=False) as tepoch:
            for batch in tepoch:
                u, pos, neg = batch
                u, pos, neg = u.to(Config.device), pos.to(Config.device), neg.to(Config.device)
                
                optimizer.zero_grad()
                
                x1 = model.forward_encoder(graphs, perturbed=True)
                x2 = model.forward_encoder(graphs, perturbed=True)
                
                # A. 推荐 BPR Loss
                u_emb, pos_emb, neg_emb = x1[u], x1[pos], x1[neg]
                pos_scores = (u_emb * pos_emb).sum(dim=1)
                neg_scores = (u_emb * neg_emb).sum(dim=1)
                bpr_loss = -torch.mean(torch.nn.functional.logsigmoid(pos_scores - neg_scores))
                
                # B. Graph SSL Loss
                unique_nodes = torch.unique(torch.cat([u, pos, neg]))
                graph_ssl_loss = model.calc_ssl_loss(x1, x2, unique_nodes)
                
                # C. Cross-Modal SSL
                cm_ssl_loss = torch.tensor(0.0, device=Config.device)
                if USE_CROSS_MODAL:
                    unique_herbs = torch.unique(torch.cat([pos, neg]))
                    cm_ssl_loss = model.calc_cross_modal_loss(x1, unique_herbs)

                # D. Property-Chem SSL
                pc_ssl_loss = torch.tensor(0.0, device=Config.device)
                if USE_PROP_CHEM_ALIGN:
                    unique_herbs = torch.unique(torch.cat([pos, neg]))
                    pc_ssl_loss = model.calc_property_chem_loss(unique_herbs)
                
                # 四个损失函数的组合完全对齐 train.py
                loss = (
                    bpr_loss
                    + Config.ssl_reg * graph_ssl_loss
                    + CROSS_MODAL_WEIGHT * cm_ssl_loss
                    + PROP_CHEM_WEIGHT * pc_ssl_loss
                )

                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                tepoch.set_postfix(loss=loss.item())
                
        avg_loss = total_loss / len(train_loader)
        if (epoch + 1) % Config.eval_interval == 0:
            print(f"Epoch {epoch+1} | Loss: {avg_loss:.4f}")
            
            results = evaluator.evaluate(
                model, 
                graphs, 
                data_pack['test_dict'], 
                data_pack['herb_indices'], 
                Config.device
            )
            
            res_str = " | ".join([f"{k}: {v:.4f}" for k, v in results.items() if 'F1' in k])
            print(f"   >> Test Metrics: {res_str}")
            
            cur_f1 = results['F1@10']
            
            if cur_f1 > best_f1:
                best_f1 = cur_f1
                no_improve_cnt = 0
                torch.save(model.state_dict(), save_path)
                print(f"   >> ⭐ New Best Model! F1@10: {best_f1:.4f}")
            else:
                no_improve_cnt += 1
                print(f"   >> No improvement. Counter: {no_improve_cnt}/{Config.patience}")
                
                if no_improve_cnt >= Config.patience:
                    print(f"\n[Early Stopping] Triggered.")
                    break

    # --- 最终评估 ---
    if os.path.exists(save_path):
        model.load_state_dict(torch.load(save_path, map_location=Config.device))
    
    final_results = evaluator.evaluate(
        model, graphs, data_pack['test_dict'], data_pack['herb_indices'], Config.device
    )

    print("\n" + "=" * 50)
    print("Final Split Multi-View (Ablation) Test Results:")
    for k in Config.top_k:
        pk = final_results.get(f'Precision@{k}', 0.0)
        rk = final_results.get(f'Recall@{k}', 0.0)
        fk = final_results.get(f'F1@{k}', 0.0)
        print(f"  P@{k}={pk:.4f}  R@{k}={rk:.4f}  F1@{k}={fk:.4f}")

if __name__ == "__main__":
    main()
