import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

# 路径配置
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
MKG_DIR = os.path.dirname(CURRENT_DIR)
DATA_ROOT = os.path.join(MKG_DIR, 'dataset', 'NEWHERB')
KGE_DIR = os.path.join(DATA_ROOT, 'kge_data')

def analyze():
    print("Loading triples...")
    files = ['train.tsv', 'dev.tsv', 'test.tsv']
    
    # 1. 统计映射
    herb_set = set()
    disease_set = set()
    
    # 2. 统计度数 (Degree)
    # herb_degree: 这个药治多少个病
    herb_degree = defaultdict(int)
    # disease_degree: 这个病由多少个药治 (关键指标!)
    disease_degree = defaultdict(int)
    
    for fname in files:
        path = os.path.join(KGE_DIR, fname)
        if not os.path.exists(path): continue
        df = pd.read_csv(path, sep='\t', header=None, names=['h', 'r', 't'])
        
        for _, row in df.iterrows():
            if row['r'] == 'treats_disease':
                h, t = row['h'], row['t']
                herb_set.add(h)
                disease_set.add(t)
                
                herb_degree[h] += 1
                disease_degree[t] += 1

    print(f"\n{'='*30}")
    print(f"数据概览 (Data Overview)")
    print(f"{'='*30}")
    print(f"Herbs (草药数): {len(herb_set)}")
    print(f"Diseases (疾病数): {len(disease_set)}")
    print(f"Ratio (病/药): {len(disease_set)/len(herb_set):.2f}")
    
    # --- 核心诊断 1: 疾病的度分布 ---
    print(f"\n[诊断 1] 疾病的度分布 (一个病由几个药治?)")
    degrees = list(disease_degree.values())
    print(f"Max degree: {np.max(degrees)}")
    print(f"Min degree: {np.min(degrees)}")
    print(f"Avg degree: {np.mean(degrees):.2f}")
    print(f"Median degree: {np.median(degrees)}")
    
    # 统计长尾
    tail_1 = sum(1 for d in degrees if d == 1)
    tail_2 = sum(1 for d in degrees if d <= 2)
    print(f"只关联 1 个药的疾病数: {tail_1} (占比 {tail_1/len(disease_set):.2%})")
    print(f"关联 <= 2 个药的疾病数: {tail_2} (占比 {tail_2/len(disease_set):.2%})")
    
    if tail_1 / len(disease_set) > 0.5:
        print("⚠️ 警告: 超过50%的疾病是长尾节点(度为1)。这会导致推荐极难进行！")

    # --- 核心诊断 2: 草药的度分布 ---
    print(f"\n[诊断 2] 草药的度分布 (一个药治多少病?)")
    h_degrees = list(herb_degree.values())
    print(f"Max degree: {np.max(h_degrees)} (Hub节点)")
    print(f"Avg degree: {np.mean(h_degrees):.2f}")
    
    # 找出 Top-5 Hub Herbs
    sorted_herbs = sorted(herb_degree.items(), key=lambda x: x[1], reverse=True)
    print("Top 5 '万金油' 草药:")
    for h, d in sorted_herbs[:5]:
        print(f"  - {h}: 治疗 {d} 种病")

if __name__ == "__main__":
    analyze()