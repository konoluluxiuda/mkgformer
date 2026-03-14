import os
import pandas as pd
import numpy as np
import torch
from scipy import sparse
from collections import defaultdict
from tqdm import tqdm
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# === 配置 ===
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
MKG_DIR = os.path.dirname(CURRENT_DIR)
DATA_ROOT = os.path.join(MKG_DIR, 'dataset', 'NEWHERB')
KGE_DIR = os.path.join(DATA_ROOT, 'kge_data')
FEATURE_DIR = os.path.join(DATA_ROOT, 'features')
# 输出到 semantic_data 文件夹，避免覆盖原版
OUTPUT_DIR = os.path.join(DATA_ROOT, 'semantic_data') 
os.makedirs(OUTPUT_DIR, exist_ok=True)

VALID_RELATIONS = {'treats_disease', 'has_component', 'has_effect', 'has_property', 'belongs_to_meridian'}
TOP_K_SEMANTIC = 10  # 每个节点连接最相似的 5 个
SIM_THRESHOLD = 0.3 # 文本相似度阈值

def build_semantic_edges(ent2id, filename, threshold, top_k):
    """读取文本文件，计算 TF-IDF 相似度并构建边"""
    path = os.path.join(FEATURE_DIR, filename)
    if not os.path.exists(path):
        print(f"Warning: {filename} not found.")
        return [], []
    
    print(f"   Processing {filename}...")
    names, texts = [], []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) >= 2:
                name, desc = parts[0], parts[1]
                if name in ent2id:
                    names.append(ent2id[name])
                    texts.append(desc)
    
    if not names: return [], []

    # TF-IDF + Cosine Similarity
    tfidf = TfidfVectorizer(max_features=2000).fit_transform(texts)
    sim_mat = cosine_similarity(tfidf) # [N, N]
    
    src, dst = [], []
    count = 0
    # Top-K 截断
    for i in range(len(names)):
        # 获取第 i 行的分数
        scores = sim_mat[i]
        # 排序索引 (降序)
        top_indices = scores.argsort()[::-1][1:top_k+1] # 排除自己(0)
        
        u = names[i]
        for idx in top_indices:
            score = scores[idx]
            if score > threshold:
                v = names[idx]
                src.append(u)
                dst.append(v)
                count += 1
    return src, dst

def process():
    print("1. Loading mappings...")
    with open(os.path.join(KGE_DIR, 'entities.txt'), 'r') as f:
        ent_lines = [l.strip() for l in f if l.strip()]
    ent2id = {name: i for i, name in enumerate(ent_lines)}
    num_nodes = len(ent_lines)

    # 关系映射 (在原有基础上增加语义关系)
    rel_type_map = {
        'treats_disease': 0, 'has_component': 1, 'has_effect': 2,
        'has_property': 3, 'belongs_to_meridian': 4
    }
    # 0-9: 基础+反向; 10: HH_Collab; 11: DD_Collab
    # 新增 -> 12: Herb_Semantic; 13: Disease_Semantic
    REL_HERB_SEM = 12
    REL_DISEASE_SEM = 13
    TOTAL_RELATIONS = 14 

    edges_src, edges_dst, edges_type = [], [], []
    disease_herb_dict = defaultdict(set)
    all_herbs = set()

    # --- A. 读取原有三元组 (基础结构) ---
    print("2. Reading structural triples...")
    files = ['train.tsv', 'dev.tsv', 'test.tsv']
    for filename in files:
        path = os.path.join(KGE_DIR, filename)
        if not os.path.exists(path): continue
        df = pd.read_csv(path, sep='\t', header=None, names=['h', 'r', 't'])
        for _, row in df.iterrows():
            h, r, t = row['h'], row['r'], row['t']
            if r not in VALID_RELATIONS: continue
            if h not in ent2id or t not in ent2id: continue
            
            h_idx, t_idx = ent2id[h], ent2id[t]
            r_id = rel_type_map[r]
            
            # 双向边
            edges_src.extend([h_idx, t_idx])
            edges_dst.extend([t_idx, h_idx])
            edges_type.extend([r_id, r_id + 5])
            
            if r == 'treats_disease':
                disease_herb_dict[t_idx].add(h_idx)
                all_herbs.add(h_idx)

    # --- B. 构建语义边 (方案一核心) ---
    print("3. Building Semantic Edges (TF-IDF)...")
    
    # 1. Herb Semantic
    hh_src, hh_dst = build_semantic_edges(ent2id, 'herb2textlong.txt', SIM_THRESHOLD, TOP_K_SEMANTIC)
    edges_src.extend(hh_src)
    edges_dst.extend(hh_dst)
    edges_type.extend([REL_HERB_SEM] * len(hh_src))
    print(f"   -> Added {len(hh_src)} Herb semantic edges.")

    # 2. Disease Semantic
    dd_src, dd_dst = build_semantic_edges(ent2id, 'disease2textlong.txt', SIM_THRESHOLD, TOP_K_SEMANTIC)
    edges_src.extend(dd_src)
    edges_dst.extend(dd_dst)
    edges_type.extend([REL_DISEASE_SEM] * len(dd_src))
    print(f"   -> Added {len(dd_src)} Disease semantic edges.")

    # (可选：这里也可以加上原本的协作图逻辑，为了控制变量，这里只展示语义图)

    # --- 保存 ---
    print(f"4. Saving to {OUTPUT_DIR}...")
    edge_index = torch.tensor([edges_src, edges_dst], dtype=torch.long)
    edge_type = torch.tensor(edges_type, dtype=torch.long)
    
    # 划分数据集
    train_data, test_data = {}, {}
    np.random.seed(42)
    for d_idx, herbs in disease_herb_dict.items():
        herbs = list(herbs)
        if len(herbs) < 2: train_data[d_idx] = herbs
        else:
            np.random.shuffle(herbs)
            split = int(len(herbs) * 0.8)
            train_data[d_idx] = herbs[:split]
            test_data[d_idx] = herbs[split:]
            
    data_dict = {
        'num_nodes': num_nodes,
        'num_relations': TOTAL_RELATIONS,
        'herb_indices': list(all_herbs),
        'train_dict': train_data, 'test_dict': test_data
    }
    torch.save(edge_index, os.path.join(OUTPUT_DIR, 'edge_index.pt'))
    torch.save(edge_type, os.path.join(OUTPUT_DIR, 'edge_type.pt'))
    torch.save(data_dict, os.path.join(OUTPUT_DIR, 'rec_data.pt'))
    print("Done!")

if __name__ == "__main__":
    process()