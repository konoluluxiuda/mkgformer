import os
import pandas as pd
import numpy as np
import torch
from scipy import sparse
from collections import defaultdict
from tqdm import tqdm

# =================================================================
# 1. 配置与路径
# =================================================================
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
MKG_DIR = os.path.dirname(CURRENT_DIR)
DATA_ROOT = os.path.join(MKG_DIR, 'dataset', 'NEWHERB')
KGE_DIR = os.path.join(DATA_ROOT, 'kge_data')
OUTPUT_DIR = os.path.join(DATA_ROOT, 'paper_graph_data')
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 核心关系
VALID_RELATIONS = {
    'treats_disease', 'has_component', 'has_effect', 
    'has_property', 'belongs_to_meridian'
}

# --- 论文核心参数 (阈值 + Top-K) ---
# 策略：Jaccard 阈值用于过滤"不相关"的，Top-K 用于过滤"过于泛化"的

# H-H: 两个药必须有 80% 重叠，且只保留最相似的 10 个
MIN_COOC_HERB = 0.8     
TOP_K_HERB = 10

# S-S: 两个病必须有 80% 重叠，且只保留最相似的 15 个
# (Top-K 在这里对抗"人参效应"非常关键)
MIN_COOC_DISEASE = 0.8   
TOP_K_DISEASE = 15

# =================================================================
# 2. 核心计算函数 (共现 + 阈值 + Top-K)
# =================================================================

def build_cooccurrence_graph(interaction_mat, global_ids, threshold, top_k, name="Graph"):
    """
    改进版: Jaccard 归一化 + Top-K 截断
    """
    num_nodes = interaction_mat.shape[0]
    print(f"   Building {name} from matrix {interaction_mat.shape}...")
    
    # 1. 计算共现 (Intersection)
    intersection = interaction_mat @ interaction_mat.T
    
    # 2. 计算度数 (用于 Jaccard 分母)
    degrees = interaction_mat.sum(axis=1).A1
    
    # 3. 转换为 COO 进行遍历
    intersection = intersection.tocoo()
    
    # 用于 Top-K 排序的临时字典: row_idx -> list of (col_idx, score)
    row_neighbors = defaultdict(list)
    
    print(f"   Calculating Jaccard Scores...")
    for i, j, val in tqdm(zip(intersection.row, intersection.col, intersection.data), total=intersection.nnz):
        if i == j: continue 
        
        # 计算 Jaccard 相似度
        union = degrees[i] + degrees[j] - val
        if union > 0:
            score = val / union
        else:
            score = 0.0
            
        # 基础阈值过滤
        if score >= threshold:
            row_neighbors[i].append((j, score))
            
    # 4. 应用 Top-K 截断并构建边列表
    src_list = []
    dst_list = []
    count = 0
    
    print(f"   Pruning to Top-{top_k}...")
    for i, neighbors in row_neighbors.items():
        # 按分数降序排列
        neighbors.sort(key=lambda x: x[1], reverse=True)
        
        # 截取前 K 个
        keep_neighbors = neighbors[:top_k]
        
        u = global_ids[i]
        for j, _ in keep_neighbors:
            v = global_ids[j]
            src_list.append(u)
            dst_list.append(v)
            count += 1
            
    print(f"   -> {name}: Generated {count} edges (Jaccard >= {threshold}, Top-{top_k})")
    return src_list, dst_list

# =================================================================
# 3. 主流程
# =================================================================
def process():
    print(f"1. Loading mappings...")
    with open(os.path.join(KGE_DIR, 'entities.txt'), 'r') as f:
        ent_lines = [l.strip() for l in f if l.strip()]
    ent2id = {name: i for i, name in enumerate(ent_lines)}
    num_nodes = len(ent_lines)
    
    # 关系映射
    rel_type_map = {
        'treats_disease': 0, 'has_component': 1, 'has_effect': 2,
        'has_property': 3, 'belongs_to_meridian': 4
    }
    
    # 论文中的图类型 ID
    REL_HH = 10
    REL_SS = 11
    TOTAL_RELATIONS = 12 

    edges_src, edges_dst, edges_type = [], [], []
    
    herb_local_map = {}
    herb_global_list = [] 
    disease_local_map = {} 
    disease_global_list = []
    
    hd_rows, hd_cols = [], []
    disease_herb_dict = defaultdict(set)
    all_herbs = set()

    print("2. Reading Triples...")
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
            
            # 基础 S-H 图
            r_id = rel_type_map[r]
            edges_src.extend([h_idx, t_idx])
            edges_dst.extend([t_idx, h_idx])
            edges_type.extend([r_id, r_id + 5]) 
            
            # 收集矩阵数据
            if r == 'treats_disease':
                disease_herb_dict[t_idx].add(h_idx)
                all_herbs.add(h_idx)
                
                if h_idx not in herb_local_map:
                    herb_local_map[h_idx] = len(herb_global_list)
                    herb_global_list.append(h_idx)
                if t_idx not in disease_local_map:
                    disease_local_map[t_idx] = len(disease_global_list)
                    disease_global_list.append(t_idx)
                
                hd_rows.append(herb_local_map[h_idx])
                hd_cols.append(disease_local_map[t_idx])

    # 构建矩阵
    num_herbs = len(herb_global_list)
    num_diseases = len(disease_global_list)
    print(f"   Matrix Dimensions: Herbs={num_herbs}, Diseases={num_diseases}")

    # H-D Matrix
    H_D_mat = sparse.csr_matrix(
        (np.ones(len(hd_rows)), (hd_rows, hd_cols)),
        shape=(num_herbs, num_diseases)
    )
    # D-H Matrix
    D_H_mat = H_D_mat.T 

    # --- 3. 构建图 (Top-K 策略) ---
    print("3. Building Paper-style Graphs...")
    
    # H-H Graph
    hh_src, hh_dst = build_cooccurrence_graph(
        H_D_mat, herb_global_list, 
        threshold=MIN_COOC_HERB, top_k=TOP_K_HERB, 
        name="H-H Graph"
    )
    edges_src.extend(hh_src)
    edges_dst.extend(hh_dst)
    edges_type.extend([REL_HH] * len(hh_src))

    # S-S Graph
    ss_src, ss_dst = build_cooccurrence_graph(
        D_H_mat, disease_global_list, 
        threshold=MIN_COOC_DISEASE, top_k=TOP_K_DISEASE, 
        name="S-S Graph"
    )
    edges_src.extend(ss_src)
    edges_dst.extend(ss_dst)
    edges_type.extend([REL_SS] * len(ss_src))

    # --- 4. 保存 ---
    print("4. Saving...")
    edge_index = torch.tensor([edges_src, edges_dst], dtype=torch.long)
    edge_type = torch.tensor(edges_type, dtype=torch.long)
    
    train_data, test_data = {}, {}
    np.random.seed(42)
    for d_idx, herbs in disease_herb_dict.items():
        herbs = list(herbs)
        if len(herbs) < 2:
            train_data[d_idx] = herbs
        else:
            np.random.shuffle(herbs)
            split = int(len(herbs) * 0.8)
            train_data[d_idx] = herbs[:split]
            test_data[d_idx] = herbs[split:]
    
    data_dict = {
        'num_nodes': num_nodes,
        'num_relations': TOTAL_RELATIONS,
        'herb_indices': list(all_herbs),
        'train_dict': train_data,
        'test_dict': test_data
    }
    
    torch.save(edge_index, os.path.join(OUTPUT_DIR, 'edge_index.pt'))
    torch.save(edge_type, os.path.join(OUTPUT_DIR, 'edge_type.pt'))
    torch.save(data_dict, os.path.join(OUTPUT_DIR, 'rec_data.pt'))
    print("Done! Top-K graph structure created.")

if __name__ == "__main__":
    process()