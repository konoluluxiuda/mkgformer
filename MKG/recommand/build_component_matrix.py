# build_component_matrix.py
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
    
    # 收集成分
    comp_set = set()
    herb_comps = defaultdict(set)
    
    print("2. Scanning components...")
    files = ['train.tsv', 'dev.tsv', 'test.tsv']
    for filename in files:
        path = os.path.join(KGE_DIR, filename)
        if not os.path.exists(path): continue
        df = pd.read_csv(path, sep='\t', header=None, names=['h', 'r', 't'])
        
        for _, row in df.iterrows():
            if row['r'] == 'has_component':
                if row['h'] in ent2id:
                    comp_name = row['t']
                    comp_set.add(comp_name)
                    herb_comps[ent2id[row['h']]].add(comp_name)

    # 建立成分 ID 映射
    # 过滤掉太冷门的成分（可选，防止维度爆炸）
    # 这里先保留全部
    comp_list = sorted(list(comp_set))
    comp2id = {name: i for i, name in enumerate(comp_list)}
    num_comps = len(comp_list)
    print(f"   Found {num_comps} unique components.")
    
    # 构建 Multi-hot 矩阵 [Num_Nodes, Num_Components]
    # 注意：如果 num_comps 很大（如7000），这个矩阵比较大，但很稀疏
    print("3. Building Multi-hot Matrix...")
    
    # 使用 torch.sparse_coo_tensor 节省内存
    indices = []
    values = []
    
    for hid, cnames in herb_comps.items():
        for cname in cnames:
            if cname in comp2id:
                indices.append([hid, comp2id[cname]])
                values.append(1.0)
                
    indices = torch.tensor(indices).t()
    values = torch.tensor(values)
    
    # 创建稀疏张量
    # 注意：在 model.py 中使用 Linear 时可能需要转为 dense，或者使用 sparse mm
    # 为了兼容现有代码的 Linear 层，如果显存够，我们可以转 dense
    # 7000 * 13000 * 4 bytes ≈ 350MB，显存完全够用
    comp_matrix = torch.sparse_coo_tensor(indices, values, (num_nodes, num_comps)).to_dense()
    
    # 保存
    save_path = os.path.join(OUTPUT_DIR, 'node_chem_multihot.pt')
    torch.save(comp_matrix, save_path)
    print(f"Done! Saved matrix {comp_matrix.shape} to {save_path}")

if __name__ == "__main__":
    main()