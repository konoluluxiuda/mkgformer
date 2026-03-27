# build_chem_fingerprint.py
import os
import torch
import pandas as pd
import numpy as np
from collections import defaultdict
from tqdm import tqdm

# ==========================================
# 1. 路径配置
# ==========================================
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
MKG_DIR = os.path.dirname(CURRENT_DIR)
DATA_ROOT = os.path.join(MKG_DIR, 'dataset', 'NEWHERB')
KGE_DIR = os.path.join(DATA_ROOT, 'kge_data')
OUTPUT_DIR = os.path.join(DATA_ROOT, 'recommendation_data')
MACCS_FILE = os.path.join(DATA_ROOT, 'features', 'component2maccs.txt')

def main():
    print("1. Loading entities...")
    with open(os.path.join(KGE_DIR, 'entities.txt'), 'r') as f:
        ent_lines = [l.strip() for l in f if l.strip()]
    ent2id = {name: i for i, name in enumerate(ent_lines)}
    num_nodes = len(ent_lines)
    
    # 记录药材包含哪些成分 (Herb Entity ID -> set of Component Names)
    herb_comps = defaultdict(set)
    
    print("2. Scanning Herb-Component relations from KGE data...")
    files = ['train.tsv', 'dev.tsv', 'test.tsv']
    for filename in files:
        path = os.path.join(KGE_DIR, filename)
        if not os.path.exists(path): continue
        df = pd.read_csv(path, sep='\t', header=None, names=['h', 'r', 't'])
        
        # 只提取 has_component 关系
        for _, row in df.iterrows():
            if row['r'] == 'has_component':
                if row['h'] in ent2id:
                    comp_name = row['t']
                    h_id = ent2id[row['h']]
                    herb_comps[h_id].add(comp_name)

    print("3. Loading MACCS Fingerprints...")
    comp2fingerprint = {}
    if not os.path.exists(MACCS_FILE):
        print(f"❌ Error: MACCS file not found at {MACCS_FILE}")
        return
        
    with open(MACCS_FILE, 'r') as f:
        for line in tqdm(f):
            parts = line.strip().split('\t')
            if len(parts) == 2:
                c_name, maccs_str = parts[0], parts[1]
                # 将 '0100...' 字符串转为 numpy 浮点数组 (通常 MACCS 是 166 维)
                fp_array = np.array([float(bit) for bit in maccs_str])
                comp2fingerprint[c_name] = fp_array
                
    maccs_dim = len(next(iter(comp2fingerprint.values())))
    print(f"   => Loaded {len(comp2fingerprint)} fingerprints, Dimension: {maccs_dim}")

    print("4. Aggregating Fingerprints for Herbs (Mean Pooling)...")
    # 初始化全节点的特征矩阵 (包含疾病、成分等其他节点，它们保持为全0)
    fingerprint_matrix = np.zeros((num_nodes, maccs_dim), dtype=np.float32)
    
    hit_counts = []
    
    # 遍历每个药材，聚合其包含的成分特征
    for h_id, c_names in herb_comps.items():
        fp_list = []
        for c_name in c_names:
            if c_name in comp2fingerprint:
                fp_list.append(comp2fingerprint[c_name])
                
        if len(fp_list) > 0:
            # 使用均值池化 (Mean Pooling) 聚合化学信息
            mean_fp = np.mean(fp_list, axis=0)
            fingerprint_matrix[h_id] = mean_fp
            hit_counts.append(len(fp_list))
            
    print(f"   => {len(hit_counts)} herbs have valid fingerprint representations.")
    if hit_counts:
        print(f"   => Average components with MACCS per herb: {np.mean(hit_counts):.2f}")

    print("5. Saving to Pytorch Tensor...")
    fp_tensor = torch.tensor(fingerprint_matrix, dtype=torch.float32)
    save_path = os.path.join(OUTPUT_DIR, 'node_chem_fingerprint.pt')
    torch.save(fp_tensor, save_path)
    
    print(f"✅ Success! Saved MACCS Matrix {fp_tensor.shape} to {save_path}")
    print("Now you can run train.py and the 'Chem Fingerprint enabled' warning will disappear!")

if __name__ == "__main__":
    main()