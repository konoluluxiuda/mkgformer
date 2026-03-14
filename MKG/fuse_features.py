# fuse_features.py (Final Integrated Version for NEWHERB)

import torch
import torch.nn as nn
import os
import json
import numpy as np
from tqdm import tqdm
from pathlib import Path

# 导入 Transformers (使用具体类以避免网络问题)
from transformers import BertTokenizer, BertConfig, RobertaTokenizer, RobertaModel

# 导入本地模型定义
from models.model import UnimoKGC
from models.modeling_clip import CLIPModel

# =================================================================
# 1. 配置与初始化
# =================================================================
@torch.no_grad()
def main():
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- 路径系统 ---
    SCRIPT_DIR = Path(__file__).parent.resolve()
    
    # 数据集路径
    DATA_ROOT = SCRIPT_DIR / "dataset" / "NEWHERB"
    KGE_DATA_DIR = DATA_ROOT / "kge_data"
    FEATURES_DIR = DATA_ROOT / "features"
    
    # 输出路径
    OUTPUT_DIR = SCRIPT_DIR / "output" / "NEWHERB"
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # 模型路径 (本地)
    MODEL_DIR = SCRIPT_DIR / "models"
    BERT_PATH = MODEL_DIR / "bert-base-chinese"
    CLIP_PATH = MODEL_DIR / "clip-vit-base-patch32"
    SMILES_PATH = MODEL_DIR / "ChemBERTa-zinc-base-v1"

    # 检查关键文件是否存在
    if not (BERT_PATH / "vocab.txt").exists():
        raise FileNotFoundError(f"BERT model not found at {BERT_PATH}")
    if not (FEATURES_DIR / "entity2textlong.txt").exists():
        raise FileNotFoundError(f"Text features not found. Please run '6_merge_text_features.py' first.")

    # =================================================================
    # 2. 加载模型
    # =================================================================
    print("Loading models...")

    # 1. UnimoKGC (用于提取文本特征)
    # 即使不使用图像，UnimoKGC 的架构也需要 CLIP 的配置来初始化
    tokenizer_bert = BertTokenizer.from_pretrained(BERT_PATH, local_files_only=True)
    config_bert = BertConfig.from_pretrained(BERT_PATH, local_files_only=True)
    config_clip = CLIPModel.from_pretrained(CLIP_PATH, local_files_only=True).config
    
    model_unimo = UnimoKGC(config_clip.vision_config, config_bert, pretrain=False)
    model_unimo.to(device).eval()

    # 2. ChemBERTa (用于提取 SMILES 特征)
    tokenizer_smiles = RobertaTokenizer.from_pretrained(SMILES_PATH, local_files_only=True)
    model_smiles = RobertaModel.from_pretrained(SMILES_PATH, local_files_only=True)
    model_smiles.to(device).eval()

    # 3. 线性投影层 (用于融合 Component 的 Text+SMILES)
    # 输入: 768(Text) + 768(SMILES) = 1536 -> 输出: 768
    embedding_dim = config_bert.hidden_size
    projection_layer = nn.Linear(embedding_dim * 2, embedding_dim)
    # 注意：这里使用随机初始化，在后续 KGE 训练中会微调生成的 Embedding
    projection_layer.to(device).eval()

    print("Models loaded successfully.")

    # =================================================================
    # 3. 加载数据映射
    # =================================================================
    print("Loading data maps...")

    # 1. 实体列表 (基准)
    with open(KGE_DATA_DIR / "entities.txt", 'r', encoding='utf-8') as f:
        entity_list = [line.strip() for line in f if line.strip()]
    
    # 2. 实体类型 (用于判断是否为 Component)
    entity2type = {}
    with open(KGE_DATA_DIR / "entity2type.txt", 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) == 2: entity2type[parts[0]] = parts[1]

    # 3. 文本描述 (统一文件)
    entity_text_map = {}
    with open(FEATURES_DIR / "entity2textlong.txt", 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split('\t', 1)
            if len(parts) == 2: entity_text_map[parts[0]] = parts[1]

    # 4. SMILES 数据
    component_smiles_map = {}
    smiles_file = FEATURES_DIR / "component2smiles.txt"
    if smiles_file.exists():
        with open(smiles_file, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) == 2: component_smiles_map[parts[0]] = parts[1]
    
    print(f"Loaded {len(entity_list)} entities.")
    print(f"Loaded {len(component_smiles_map)} SMILES strings.")

    # =================================================================
    # 4. 特征提取主循环
    # =================================================================
    final_embeddings = []
    
    # 固定的全零图片张量 (用于禁用图像模态)
    zero_pixel_values = torch.zeros(1, 3, 224, 224).to(device)

    print("Starting feature extraction and fusion...")
    
    for entity_name in tqdm(entity_list, desc="Fusing Features"):
        
        # --- A. 获取文本 Embedding (基础) ---
        # 如果没有详细描述，使用实体名兜底
        text_str = entity_text_map.get(entity_name, entity_name)
        
        text_inputs = tokenizer_bert(
            text_str, 
            return_tensors="pt", 
            max_length=128, 
            padding="max_length", 
            truncation=True
        ).to(device)
        
        # 获取 BERT [CLS] embedding
        # UnimoKGC.encoder 返回的是 BaseModelOutput
        unimo_out = model_unimo.encoder(
            input_ids=text_inputs.input_ids,
            attention_mask=text_inputs.attention_mask,
            pixel_values=zero_pixel_values, # 强制无图
            return_dict=True
        )
        # [1, 768]
        text_emb = unimo_out.last_hidden_state[:, 0, :]

        # --- B. 判断融合策略 ---
        # 只有当实体是 Component 且我们有它的 SMILES 时，才进行融合
        etype = entity2type.get(entity_name)
        
        if (etype == 'Component' or etype == 'Chemical') and (entity_name in component_smiles_map):
            # 获取 SMILES Embedding
            smiles_str = component_smiles_map[entity_name]
            smiles_inputs = tokenizer_smiles(
                smiles_str,
                return_tensors="pt",
                padding="max_length",
                truncation=True,
                max_length=128
            ).to(device)
            
            smiles_out = model_smiles(**smiles_inputs)
            # [1, 768]
            smiles_emb = smiles_out.last_hidden_state[:, 0, :]
            
            # 拼接 [1, 1536]
            cat_emb = torch.cat([text_emb, smiles_emb], dim=1)
            
            # 投影回 [1, 768]
            final_emb = projection_layer(cat_emb)
            
        else:
            # 其他情况 (Herb, Protein, 或无SMILES的成分) -> 仅使用文本
            final_emb = text_emb
        
        # 转为 numpy 并存入列表
        final_embeddings.append(final_emb.squeeze().cpu().numpy())

    # =================================================================
    # 5. 保存结果
    # =================================================================
    print("Saving final matrix...")
    
    # 转换为大矩阵 [N_entities, 768]
    embedding_matrix = np.array(final_embeddings)
    
    save_path = OUTPUT_DIR / "final_entity_embeddings_text_smiles.npy"
    np.save(save_path, embedding_matrix)
    
    print("\n" + "="*30)
    print("✅ Feature Fusion Complete!")
    print(f"Processed entities: {len(entity_list)}")
    print(f"Matrix shape: {embedding_matrix.shape}")
    print(f"Saved to: {save_path}")
    print("="*30 + "\n")

if __name__ == "__main__":
    main()