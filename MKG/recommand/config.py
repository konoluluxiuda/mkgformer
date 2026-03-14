# config.py
import torch
import os

class Config:
    # ------------------ 路径配置 ------------------
    CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
    MKG_DIR = os.path.dirname(CURRENT_DIR)
    DATA_ROOT = os.path.join(MKG_DIR, 'dataset', 'NEWHERB')
    REC_DATA_DIR = os.path.join(DATA_ROOT, 'recommendation_data')
    
    # [新增] 属性矩阵路径
    ATTR_PATH = os.path.join(REC_DATA_DIR, 'node_attributes.pt')

    # [修改] 注释掉或设为 None，不再加载特征
    FEATURE_PATH = None 
    
    MODEL_SAVE_PATH = os.path.join(CURRENT_DIR, 'checkpoints')
    os.makedirs(MODEL_SAVE_PATH, exist_ok=True)

    # ------------------ 训练参数 ------------------
    device = "cuda" if torch.cuda.is_available() else "cpu"
    seed = 42
    
    epochs = 500
    batch_size = 1024
    lr = 1e-3
    weight_decay = 1e-5
    
    # 早停参数
    eval_interval = 2
    patience = 25          # 稍微给多一点耐心
    
    # ------------------ 模型参数 ------------------
    # [修改] 回归基础维度
    input_dim = 128       # 随机 Embedding 的维度
    attr_dim = 64         # [新增] 属性投影后的维度
    hidden_dim = 128      # GCN 隐层维度
    dropout = 0.2         # 建议 0.3 (Baseline 配置)
    
    # ------------------ SSL 参数 ------------------
    ssl_temp = 0.2
    ssl_reg = 0.05         # 可以尝试微调这个 (0.05 ~ 0.2)
    edge_drop_rate = 0.1

    # ------------------ 评估参数 ------------------
    top_k = [5, 10, 20, 50]