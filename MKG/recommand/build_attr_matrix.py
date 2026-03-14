# build_attr_matrix.py
import os
import torch
import pandas as pd
import numpy as np
from collections import defaultdict

# 路径配置
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
MKG_DIR = os.path.dirname(CURRENT_DIR)
DATA_ROOT = os.path.join(MKG_DIR, 'dataset', 'NEWHERB')
KGE_DIR = os.path.join(DATA_ROOT, 'kge_data')
OUTPUT_DIR = os.path.join(DATA_ROOT, 'recommendation_data')

def main():
    print("1. Loading entities...")
    with open(os.path.join(KGE_DIR, 'entities.txt'), 'r') as f:
        ent_lines = [l.strip() for l in f if l.strip()]
    ent2id = {name: i for i, name in enumerate(ent_lines)}
    num_nodes = len(ent_lines)
    
    # 定义需要提取的属性关系
    ATTR_RELATIONS = {'has_property', 'belongs_to_meridian'} # 也可以加上 'has_effect'
    
    # 收集所有属性实体
    attr_set = set()
    # 记录节点拥有的属性
    node_attrs = defaultdict(set)
    
    print("2. Scanning attributes...")
    files = ['train.tsv', 'dev.tsv', 'test.tsv']
    for filename in files:
        path = os.path.join(KGE_DIR, filename)
        if not os.path.exists(path): continue
        df = pd.read_csv(path, sep='\t', header=None, names=['h', 'r', 't'])
        
        for _, row in df.iterrows():
            h, r, t = row['h'], row['r'], row['t']
            
            if r in ATTR_RELATIONS:
                if h in ent2id:
                    # t 是属性名 (如 "温", "脾经")
                    attr_set.add(t)
                    node_attrs[ent2id[h]].add(t)

    # 建立属性 ID 映射
    attr_list = sorted(list(attr_set))
    attr2id = {name: i for i, name in enumerate(attr_list)}
    num_attrs = len(attr_list)
    
    print(f"   Found {num_attrs} unique attributes (Properties/Meridians).")
    
    # 构建 Multi-hot 矩阵
    # Shape: [Num_Nodes, Num_Attributes]
    print("3. Building Multi-hot Matrix...")
    attr_matrix = np.zeros((num_nodes, num_attrs), dtype=np.float32)
    
    for nid, attrs in node_attrs.items():
        for a in attrs:
            if a in attr2id:
                attr_matrix[nid, attr2id[a]] = 1.0
                
    # 保存
    save_path = os.path.join(OUTPUT_DIR, 'node_attributes.pt')
    torch.save(torch.from_numpy(attr_matrix), save_path)
    print(f"Done! Saved matrix {attr_matrix.shape} to {save_path}")

if __name__ == "__main__":
    main()