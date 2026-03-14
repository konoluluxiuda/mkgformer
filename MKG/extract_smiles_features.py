# extract_smiles_features.py

import torch
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm
import numpy as np
import os
from pathlib import Path

# =================================================================
# 1. 配置
# =================================================================
@torch.no_grad()
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- 模型配置 ---
    # ChemBERTa 是一个在大量 SMILES 上预训练的 RoBERTa 模型
   # ✅ 使用本地路径
    SMILES_MODEL_PATH = "models/ChemBERTa-zinc-base-v1"

    # --- 路径配置 ---
    current_dir = Path(__file__).parent.resolve()
    data_dir = current_dir / "dataset" / "HERB"
    output_dir = current_dir / "output" / "HERB"
    os.makedirs(output_dir, exist_ok=True)
    
    # 输入文件 (由 fetch_smiles.py 生成)
    input_smiles_file = data_dir / "Tools/component2smiles_llm.txt"
    
    # 输出文件
    output_embedding_file = output_dir / "component_smiles_embeddings.npy"
    output_map_file = output_dir / "component_smiles_map.txt"

    if not input_smiles_file.is_file():
        raise FileNotFoundError(
            f"Input file '{input_smiles_file}' not found. \n"
            "Please run 'fetch_smiles.py' first and ensure it has generated some results."
        )

    # =================================================================
    # 2. 加载 ChemBERTa 模型和 Tokenizer
    # =================================================================
    print(f"Loading ChemBERTa model from: {SMILES_MODEL_PATH}")
    tokenizer = AutoTokenizer.from_pretrained(SMILES_MODEL_PATH)
    model = AutoModel.from_pretrained(SMILES_MODEL_PATH)
    
    model.to(device)
    model.eval()

    # =================================================================
    # 3. 读取 SMILES 数据
    # =================================================================
    smiles_map = {}
    with open(input_smiles_file, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) == 2:
                # key: component name, value: smiles string
                smiles_map[parts[0]] = parts[1]
    
    if not smiles_map:
        print("WARNING: No SMILES data found in the input file. Exiting.")
        return

    print(f"Found {len(smiles_map)} components with SMILES strings to encode.")
    
    component_list = list(smiles_map.keys())
    smiles_list = list(smiles_map.values())
    
    # =================================================================
    # 4. 批量编码 SMILES 并提取特征
    # =================================================================
    batch_size = 64 # 根据你的 GPU 显存调整
    all_embeddings = []

    print("Encoding SMILES strings and extracting embeddings...")
    for i in tqdm(range(0, len(smiles_list), batch_size), desc="Encoding SMILES"):
        batch_smiles = smiles_list[i : i + batch_size]
        
        # 使用 tokenizer 对一个批次的 SMILES 进行编码
        inputs = tokenizer(
            batch_smiles, 
            return_tensors="pt", 
            padding=True, 
            truncation=True, 
            max_length=128 # SMILES 字符串通常不会很长
        ).to(device)
        
        # 模型前向传播
        outputs = model(**inputs)
        
        # 提取 [CLS] token 的 embedding
        # outputs.last_hidden_state 的形状是 [batch_size, seq_len, hidden_size]
        cls_embeddings = outputs.last_hidden_state[:, 0, :]
        
        all_embeddings.append(cls_embeddings.cpu().numpy())

    # 将所有批次的结果合并成一个大的 NumPy 数组
    embedding_matrix = np.concatenate(all_embeddings, axis=0)

    # =================================================================
    # 5. 保存结果
    # =================================================================
    # 保存 embedding 矩阵
    np.save(output_embedding_file, embedding_matrix)
    
    # 保存 component name 的顺序，确保与 embedding 矩阵的行一一对应
    with open(output_map_file, "w", encoding="utf-8") as f:
        for name in component_list:
            f.write(name + "\n")

    print("\n" + "="*30)
    print("SMILES Feature Extraction Complete!")
    print(f"Processed {embedding_matrix.shape[0]} components.")
    print(f"Embedding matrix shape: {embedding_matrix.shape}")
    print(f"Embeddings saved to: {output_embedding_file}")
    print(f"Component map saved to: {output_map_file}")
    print("="*30 + "\n")


if __name__ == "__main__":
    main()