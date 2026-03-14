# build_deep_chem.py
import os
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from collections import defaultdict
from transformers import AutoTokenizer, AutoModel

# =================================================================
# 1. 配置路径 (完全适配你的本地环境)
# =================================================================
SCRIPT_DIR = Path(__file__).parent.resolve()
PROJECT_ROOT = SCRIPT_DIR.parent.parent # mkgformer root if needed
MKG_DIR = SCRIPT_DIR.parent
DATA_ROOT = MKG_DIR / "dataset" / "NEWHERB"

KGE_DIR = DATA_ROOT / "kge_data"
FEATURE_DIR = DATA_ROOT / "features"
OUTPUT_DIR = DATA_ROOT / "recommendation_data"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 模型路径
# 假设你的目录结构是: mkgformer/MKG/models/bert-base-chinese
MODEL_DIR = MKG_DIR / "models"
BERT_PATH = MODEL_DIR / "bert-base-chinese"
SMILES_PATH = MODEL_DIR / "ChemBERTa-zinc-base-v1"

# 设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_local_models():
    print(f"Loading models from {MODEL_DIR}...")
    
    # 1. Load ChemBERTa
    try:
        tokenizer_smiles = AutoTokenizer.from_pretrained(SMILES_PATH, local_files_only=True)
        model_smiles = AutoModel.from_pretrained(SMILES_PATH, local_files_only=True).to(device)
        print("✅ ChemBERTa loaded.")
    except Exception as e:
        print(f"❌ Failed to load ChemBERTa: {e}")
        return None, None, None, None

    # 2. Load BERT-Chinese
    try:
        tokenizer_bert = AutoTokenizer.from_pretrained(BERT_PATH, local_files_only=True)
        model_bert = AutoModel.from_pretrained(BERT_PATH, local_files_only=True).to(device)
        print("✅ BERT-Chinese loaded.")
    except Exception as e:
        print(f"❌ Failed to load BERT: {e}")
        return None, None, None, None

    return tokenizer_smiles, model_smiles, tokenizer_bert, model_bert

@torch.no_grad()
def get_embedding(text, tokenizer, model):
    """通用提取函数: 输入文本/SMILES -> 输出 [768] 向量"""
    if not text or pd.isna(text):
        return np.zeros(768, dtype=np.float32)
        
    inputs = tokenizer(text, return_tensors="pt", max_length=512, truncation=True, padding=True).to(device)
    outputs = model(**inputs)
    # 取 [CLS] token (index 0) 作为句向量
    emb = outputs.last_hidden_state[:, 0, :].cpu().numpy()[0]
    return emb

def main():
    # 1. 只加载 BERT-Chinese
    print(f"Loading BERT from {BERT_PATH}...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(BERT_PATH, local_files_only=True)
        model = AutoModel.from_pretrained(BERT_PATH, local_files_only=True).to(device)
    except Exception as e:
        print(f"Error loading BERT: {e}")
        return

    # 2. 加载实体
    with open(KGE_DIR / "entities.txt", 'r', encoding='utf-8') as f:
        ent_lines = [l.strip() for l in f if l.strip()]
    ent2id = {name: i for i, name in enumerate(ent_lines)}
    num_nodes = len(ent_lines)

    # 3. 只加载中文描述
    print("Loading component texts...")
    comp_text = {}
    txt_file = FEATURE_DIR / "component2textlong.txt"
    if txt_file.exists():
        with open(txt_file, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) >= 2: 
                    # name -> text
                    comp_text[parts[0]] = parts[1]
    else:
        print("Error: component2textlong.txt not found")
        return

    # 4. 建立 Herb -> Component 映射
    print("Mapping Herbs to Components...")
    herb_to_components = defaultdict(set)
    files = ['train.tsv', 'dev.tsv', 'test.tsv']
    for fname in files:
        path = KGE_DIR / fname
        if not path.exists(): continue
        df = pd.read_csv(path, sep='\t', header=None, names=['h', 'r', 't'])
        for _, row in df.iterrows():
            if row['r'] == 'has_component':
                if row['h'] in ent2id:
                    herb_to_components[row['h']].add(row['t'])

    # 5. 编码 (只处理有文本的)
    print("Encoding Components (Text Only)...")
    comp_emb_cache = {}
    
    # 收集所有涉及且有文本的 component
    unique_comps = set()
    for comps in herb_to_components.values():
        unique_comps.update(comps)
        
    for name in tqdm(unique_comps, desc="BERT Encoding"):
        if name in comp_text:
            # 提取特征
            inputs = tokenizer(comp_text[name], return_tensors="pt", max_length=128, truncation=True, padding=True).to(device)
            with torch.no_grad():
                outputs = model(**inputs)
            emb = outputs.last_hidden_state[:, 0, :].cpu().numpy()[0]
            comp_emb_cache[name] = emb
        else:
            # 没有文本描述的成分，用全0填充
            comp_emb_cache[name] = np.zeros(768, dtype=np.float32)

    # 6. 聚合
    print("Aggregating to Herbs...")
    chem_matrix = np.zeros((num_nodes, 768), dtype=np.float32)
    cnt = 0
    for herb_name, comp_names in herb_to_components.items():
        hid = ent2id[herb_name]
        embs = []
        for c in comp_names:
            embs.append(comp_emb_cache[c]) # 取不到的在上面已经设为0了
            
        if embs:
            # Mean Pooling
            herb_vec = np.mean(np.stack(embs), axis=0)
            chem_matrix[hid] = herb_vec
            cnt += 1

    save_path = OUTPUT_DIR / "node_chem_dense.pt" # 覆盖原文件
    torch.save(torch.from_numpy(chem_matrix), save_path)
    print(f"Done! Saved text-only matrix to {save_path}")

if __name__ == "__main__":
    main()