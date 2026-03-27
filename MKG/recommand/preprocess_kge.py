import os
import pandas as pd
import numpy as np
import torch
from scipy import sparse
from collections import defaultdict
from tqdm import tqdm
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# =================================================================
# 1. 配置与路径
# =================================================================
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
MKG_DIR = os.path.dirname(CURRENT_DIR)
DATA_ROOT = os.path.join(MKG_DIR, 'dataset', 'NEWHERB')
KGE_DIR = os.path.join(DATA_ROOT, 'kge_data')
OUTPUT_DIR = os.path.join(DATA_ROOT, 'recommendation_data')
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 核心关系 (用于构建基础异构图)
VALID_RELATIONS = {
    'treats_disease', 'has_component', 'has_effect', 
    'has_property', 'belongs_to_meridian'
}

# --- 图构建超参数 ---
MAX_NEIGHBORS = 15      # Top-K 限制：每个节点最多保留 15 个协作邻居
HERB_SIM_TH = 0.2       # 属性相似度阈值 (Jaccard)
MIN_COOC_FREQ = 2       # 最小共现频率 (至少共享 1 个对象)

# =================================================================
# 2. 核心计算函数 (矩阵加速版)
# =================================================================

def get_topk_edges_from_matrix(interaction_mat, global_ids, threshold, top_k, mode='similarity'):
    """
    输入: 稀疏矩阵 (N x Features)
    输出: 边的列表 (Source, Target)
    逻辑: 计算 M @ M.T，得到相似度/共现矩阵，然后取 Top-K
    """
    num_nodes = interaction_mat.shape[0]
    
    # 1. 计算关联矩阵 (N x N)
    # 结果矩阵 entry [i, j] 表示 i 和 j 共享的 Feature 数量 (共现频次)
    print(f"   > Computing {mode} matrix ({num_nodes}x{num_nodes})...")
    sim_matrix = interaction_mat @ interaction_mat.T
    
    # 如果是 Jaccard 相似度模式，需要除以 Union
    if mode == 'jaccard':
        # Union = deg[i] + deg[j] - Intersection
        degrees = interaction_mat.sum(axis=1).A1
        # 为了高效，我们将稀疏矩阵转为 COO 遍历
        sim_matrix = sim_matrix.tocoo()
    else:
        # Co-occurrence 模式，直接用频次
        sim_matrix = sim_matrix.tocoo()
    
    src_list = []
    dst_list = []
    
    # 2. 提取边 (使用字典暂存以进行排序)
    # row_idx -> list of (col_idx, score)
    neighbors = defaultdict(list)
    
    for i, j, val in zip(sim_matrix.row, sim_matrix.col, sim_matrix.data):
        if i == j: continue # 跳过自环
        
        score = 0.0
        if mode == 'jaccard':
            union = degrees[i] + degrees[j] - val
            if union > 0:
                score = val / union
        else:
            # frequency mode
            score = val
            
        if score >= threshold:
            neighbors[i].append((j, score))
            
    # 3. Top-K 截断
    count = 0
    for i, n_list in neighbors.items():
        # 按分数降序
        n_list.sort(key=lambda x: x[1], reverse=True)
        # 截断
        keep = n_list[:top_k]
        
        # 映射回 Global ID
        u_global = global_ids[i]
        for j, _ in keep:
            v_global = global_ids[j]
            src_list.append(u_global)
            dst_list.append(v_global)
            count += 1
            
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
    REV_OFFSET = 5 
    REL_HERB_COLLAB = 10     # 混合的草药协作边
    REL_DISEASE_COLLAB = 11  # 疾病协作边
    TOTAL_RELATIONS = 12 

    # 存储基础边
    edges_src, edges_dst, edges_type = [], [], []
    
    # 临时存储用于构建矩阵的数据
    # Herb-Attribute (用于相似度)
    h_attr_rows, h_attr_cols = [], [] 
    herb_local_map = {}   # Herb Global ID -> Matrix Row Index
    herb_global_list = [] # Matrix Row Index -> Herb Global ID
    attr_map = {}         # Attribute Global ID -> Matrix Col Index
    
    # Herb-Disease (用于共现)
    # 我们需要构建两个矩阵:
    # 1. H-D Matrix: Rows=Herbs, Cols=Diseases (用于算 H-H 共现)
    # 2. D-H Matrix: Rows=Diseases, Cols=Herbs (用于算 D-D 共现)
    
    disease_local_map = {} # Disease Global ID -> Matrix Row Index
    disease_global_list = []
    
    # 推荐数据集
    disease_herb_dict = defaultdict(set)

    print("2. Reading Triples...")
    files = ['train.tsv', 'dev.tsv', 'test.tsv']
    
    for filename in files:
        path = os.path.join(KGE_DIR, filename)
        if not os.path.exists(path): continue
        df = pd.read_csv(path, sep='\t', header=None, names=['h', 'r', 't'])
        
        for _, row in tqdm(df.iterrows(), total=len(df), desc=filename):
            h, r, t = row['h'], row['r'], row['t']
            
            if r not in VALID_RELATIONS: continue
            if h not in ent2id or t not in ent2id: continue
            
            h_idx, t_idx = ent2id[h], ent2id[t]
            
            # --- 构建基础图 ---
            r_id = rel_type_map[r]
            edges_src.extend([h_idx, t_idx])
            edges_dst.extend([t_idx, h_idx])
            edges_type.extend([r_id, r_id + REV_OFFSET])
            
            # --- 收集数据用于高级图构建 ---
            if r == 'treats_disease':
                # Herb -> Disease
                disease_herb_dict[t_idx].add(h_idx)
                
                # 注册 Herb
                if h_idx not in herb_local_map:
                    herb_local_map[h_idx] = len(herb_global_list)
                    herb_global_list.append(h_idx)
                # 注册 Disease
                if t_idx not in disease_local_map:
                    disease_local_map[t_idx] = len(disease_global_list)
                    disease_global_list.append(t_idx)
            
            else:
                # Herb -> Attributes (Component, Property, etc.)
                if h_idx not in herb_local_map:
                    herb_local_map[h_idx] = len(herb_global_list)
                    herb_global_list.append(h_idx)
                
                if t_idx not in attr_map:
                    attr_map[t_idx] = len(attr_map)
                
                # 记录 Herb-Attr 关系
                h_attr_rows.append(herb_local_map[h_idx])
                h_attr_cols.append(attr_map[t_idx])

    # --- 构建矩阵 ---
    num_herbs = len(herb_global_list)
    num_diseases = len(disease_global_list)
    num_attrs = len(attr_map)
    
    print(f"   Identified {num_herbs} Herbs, {num_diseases} Diseases.")

    # 1. Herb-Attribute Matrix (Sparse)
    H_A_mat = sparse.csr_matrix(
        (np.ones(len(h_attr_rows)), (h_attr_rows, h_attr_cols)),
        shape=(num_herbs, num_attrs)
    )
    
    # 2. Herb-Disease Matrix (Sparse)
    # 需要重新遍历 disease_herb_dict 来构建
    hd_rows, hd_cols = [], []
    for d_global, herbs in disease_herb_dict.items():
        if d_global not in disease_local_map: continue
        d_local = disease_local_map[d_global]
        
        for h_global in herbs:
            if h_global in herb_local_map:
                h_local = herb_local_map[h_global]
                # H-D Matrix: Row=Herb, Col=Disease
                hd_rows.append(h_local)
                hd_cols.append(d_local)

    H_D_mat = sparse.csr_matrix(
        (np.ones(len(hd_rows)), (hd_rows, hd_cols)),
        shape=(num_herbs, num_diseases)
    )
    
    # 3. Disease-Herb Matrix (Transposed)
    # Row=Disease, Col=Herb
    D_H_mat = H_D_mat.T 

    # --- 构建高级图 ---
    print("3. Building Collaborative Graphs...")
    
    # Task A: Herb-Herb (混合模式: 相似度 OR 共现)
    # A1. 基于属性的 Jaccard 边
    hh_sim_src, hh_sim_dst = get_topk_edges_from_matrix(
        H_A_mat, herb_global_list, threshold=HERB_SIM_TH, top_k=MAX_NEIGHBORS, mode='jaccard'
    )
    print(f"   -> Herb-Herb (Attribute): {len(hh_sim_src)} edges")
    
    # A2. 基于疾病的共现边 (频率)
    hh_freq_src, hh_freq_dst = get_topk_edges_from_matrix(
        H_D_mat, herb_global_list, threshold=MIN_COOC_FREQ, top_k=MAX_NEIGHBORS, mode='frequency'
    )
    print(f"   -> Herb-Herb (Co-occurrence): {len(hh_freq_src)} edges")
    
    # 合并去重
    hh_edges = set(zip(hh_sim_src, hh_sim_dst)) | set(zip(hh_freq_src, hh_freq_dst))
    print(f"   -> Herb-Herb (Merged): {len(hh_edges)} unique edges")
    
    for u, v in hh_edges:
        edges_src.append(u)
        edges_dst.append(v)
        edges_type.append(REL_HERB_COLLAB)

    # Task B: Disease-Disease (纯共现)
    dd_src, dd_dst = get_topk_edges_from_matrix(
        D_H_mat, disease_global_list, threshold=MIN_COOC_FREQ, top_k=MAX_NEIGHBORS, mode='frequency'
    )
    print(f"   -> Disease-Disease (Co-occurrence): {len(dd_src)} edges")
    
    for u, v in zip(dd_src, dd_dst):
        edges_src.append(u)
        edges_dst.append(v)
        edges_type.append(REL_DISEASE_COLLAB)

    # --- 保存 ---
    print("4. Saving...")
    edge_index = torch.tensor([edges_src, edges_dst], dtype=torch.long)
    edge_type = torch.tensor(edges_type, dtype=torch.long)
    
    # 划分数据集 (每个 Disease 留 20% Herb 测试)
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
        'herb_indices': herb_global_list,
        'train_dict': train_data,
        'test_dict': test_data
    }
    
    torch.save(edge_index, os.path.join(OUTPUT_DIR, 'edge_index.pt'))
    torch.save(edge_type, os.path.join(OUTPUT_DIR, 'edge_type.pt'))
    torch.save(data_dict, os.path.join(OUTPUT_DIR, 'rec_data.pt'))
    print("Done!")

if __name__ == "__main__":
    process()