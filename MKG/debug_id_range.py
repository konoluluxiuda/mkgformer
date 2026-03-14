import pandas as pd
from pathlib import Path

DATA_DIR = Path("dataset/NEWHERB/kge_data")

def check_ids():
    print(">>> 检查 ID 范围...")
    
    # 1. 读取实体和关系数量
    with open(DATA_DIR / "entities.txt", 'r') as f:
        n_ent = len([l for l in f if l.strip()])
    with open(DATA_DIR / "relations.txt", 'r') as f:
        n_rel = len([l for l in f if l.strip()])
        
    print(f"   Entities.txt count: {n_ent}")
    print(f"   Relations.txt count: {n_rel}")
    
    # 2. 读取映射
    # 这里我们重新构建映射，确保和训练脚本的逻辑一致
    # 注意：train_kge_newherb.py 是重新读取 entities.txt 来构建 ent2ix 的
    # 所以它的 ent2ix 最大 ID 是 n_ent - 1
    
    entities = []
    with open(DATA_DIR / "entities.txt", 'r') as f:
        entities = [l.strip() for l in f if l.strip()]
    ent2ix = {e: i for i, e in enumerate(entities)}
    
    relations = []
    with open(DATA_DIR / "relations.txt", 'r') as f:
        relations = [l.strip() for l in f if l.strip()]
    rel2ix = {r: i for i, r in enumerate(relations)}

    # 3. 检查 train.tsv 中的实体和关系是否都在映射表中
    print("\n>>> 扫描 train.tsv ...")
    df = pd.read_csv(DATA_DIR / "train.tsv", sep='\t', header=None, names=['h', 'r', 't'])
    
    invalid_h = []
    invalid_t = []
    invalid_r = []
    
    for i, row in df.iterrows():
        if row['h'] not in ent2ix: invalid_h.append(row['h'])
        if row['t'] not in ent2ix: invalid_t.append(row['t'])
        if row['r'] not in rel2ix: invalid_r.append(row['r'])
        
    if invalid_h or invalid_t or invalid_r:
        print("❌ 发现非法 ID！")
        print(f"   非法头实体数: {len(invalid_h)} (示例: {invalid_h[:3]})")
        print(f"   非法尾实体数: {len(invalid_t)} (示例: {invalid_t[:3]})")
        print(f"   非法关系数:   {len(invalid_r)} (示例: {invalid_r[:3]})")
        print("\n结论: train.tsv 包含未在 entities/relations.txt 中定义的元素。请重新运行 3_build_kge_from_local.py")
    else:
        print("✅ train.tsv 数据 ID 检查通过。所有实体和关系都在映射表中。")
        
    # 4. 检查特征文件的维度
    import numpy as np
    feat_path = Path("output/HERB/final_entity_embeddings_text_smiles.npy")
    if feat_path.exists():
        feats = np.load(feat_path)
        print(f"\n>>> 特征文件检查:")
        print(f"   Shape: {feats.shape}")
        if feats.shape[0] != n_ent:
            print(f"❌ 严重错误: 特征数量 ({feats.shape[0]}) 与 实体数量 ({n_ent}) 不一致！")
            print("   这将导致 Embedding 层初始化错误。")
        else:
            print("✅ 特征数量与实体数量一致。")

if __name__ == "__main__":
    check_ids()