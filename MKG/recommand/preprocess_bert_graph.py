import os
import torch
import pandas as pd
import numpy as np
from tqdm import tqdm
from pathlib import Path
from collections import defaultdict
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics.pairwise import cosine_similarity

# =================================================================
# 1. 配置
# =================================================================
SCRIPT_DIR = Path(__file__).parent.resolve()
MKG_DIR = SCRIPT_DIR.parent
DATA_ROOT = MKG_DIR / "dataset" / "NEWHERB"

KGE_DIR = DATA_ROOT / "kge_data"
FEATURE_DIR = DATA_ROOT / "features"
# 输出到 semantic_data (与之前的 TF-IDF 版本区分，或者覆盖)
OUTPUT_DIR = DATA_ROOT / "semantic_data" 
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 模型路径
MODEL_DIR = MKG_DIR / "models"
BERT_PATH = MODEL_DIR / "bert-base-chinese"

# 构图超参数
# BERT 的相似度通常普遍较高，阈值建议设高一点
SIM_THRESHOLD = 0.98 
TOP_K = 10
BATCH_SIZE = 64

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =================================================================
# 2. 辅助函数
# =================================================================

def load_bert():
    print(f"Loading BERT from {BERT_PATH}...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(BERT_PATH, local_files_only=True)
        model = AutoModel.from_pretrained(BERT_PATH, local_files_only=True).to(device)
        model.eval()
        return tokenizer, model
    except Exception as e:
        print(f"❌ Error loading BERT: {e}")
        exit()

def get_bert_embeddings(text_list, tokenizer, model):
    """批量提取 BERT 特征"""
    embs = []
    # 分批处理防止显存爆炸
    for i in tqdm(range(0, len(text_list), BATCH_SIZE), desc="BERT Encoding"):
        batch_texts = text_list[i : i + BATCH_SIZE]
        
        # 替换空文本，防止报错
        batch_texts = [t if isinstance(t, str) and len(t) > 0 else "未知" for t in batch_texts]
        
        inputs = tokenizer(
            batch_texts, 
            return_tensors="pt", 
            padding=True, 
            truncation=True, 
            max_length=256
        ).to(device)
        
        with torch.no_grad():
            outputs = model(**inputs)
        
        # 取 [CLS] token
        batch_emb = outputs.last_hidden_state[:, 0, :].cpu().numpy()
        embs.append(batch_emb)
        
    return np.vstack(embs)

def build_edges_from_embeddings(embeddings, global_ids, threshold, top_k):
    """
    输入: [N, 768] 的向量矩阵
    输出: src_list, dst_list
    """
    if len(embeddings) == 0:
        return [], []
        
    print(f"   Computing Cosine Similarity ({len(embeddings)} nodes)...")
    # 计算余弦相似度矩阵 [N, N]
    sim_matrix = cosine_similarity(embeddings)
    
    # 将对角线(自己和自己)设为 -1，防止被选中
    np.fill_diagonal(sim_matrix, -1)
    
    src_list, dst_list = [], []
    count = 0
    
    print(f"   Filtering edges (Threshold={threshold}, TopK={top_k})...")
    for i in range(len(embeddings)):
        # 获取第 i 个节点的所有相似度
        scores = sim_matrix[i]
        
        # 获取 Top-K 的索引 (降序)
        top_indices = scores.argsort()[::-1][:top_k]
        
        u = global_ids[i]
        
        for idx in top_indices:
            score = scores[idx]
            if score > threshold:
                v = global_ids[idx]
                src_list.append(u)
                dst_list.append(v)
                count += 1
                
    return src_list, dst_list

# =================================================================
# 3. 主流程
# =================================================================

def main():
    # 1. 加载模型
    tokenizer, model = load_bert()
    
    # 2. 加载实体
    print("Loading entities...")
    with open(KGE_DIR / "entities.txt", 'r', encoding='utf-8') as f:
        ent_lines = [l.strip() for l in f if l.strip()]
    ent2id = {name: i for i, name in enumerate(ent_lines)}
    num_nodes = len(ent_lines)

    # 3. 读取文本数据
    print("Reading text descriptions...")
    
    # 读取 Herb 文本
    herb_names = []
    herb_texts = []
    herb_global_ids = []
    
    with open(FEATURE_DIR / "herb2textlong.txt", 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) >= 2 and parts[0] in ent2id:
                herb_names.append(parts[0])
                herb_texts.append(parts[1])
                herb_global_ids.append(ent2id[parts[0]])
                
    # 读取 Disease 文本
    disease_names = []
    disease_texts = []
    disease_global_ids = []
    
    with open(FEATURE_DIR / "disease2textlong.txt", 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) >= 2 and parts[0] in ent2id:
                disease_names.append(parts[0])
                disease_texts.append(parts[1])
                disease_global_ids.append(ent2id[parts[0]])

    # 4. BERT 编码
    print(f"Encoding {len(herb_texts)} Herbs...")
    herb_embs = get_bert_embeddings(herb_texts, tokenizer, model)
    
    print(f"Encoding {len(disease_texts)} Diseases...")
    disease_embs = get_bert_embeddings(disease_texts, tokenizer, model)

    # 5. 构建图结构
    # 基础关系映射
    rel_type_map = {
        'treats_disease': 0, 'has_component': 1, 'has_effect': 2,
        'has_property': 3, 'belongs_to_meridian': 4
    }
    # 新增语义关系 ID
    REL_HERB_SEM = 10
    REL_DISEASE_SEM = 11
    TOTAL_RELATIONS = 12
    
    edges_src, edges_dst, edges_type = [], [], []
    disease_herb_dict = defaultdict(set)
    all_herbs = set()
    VALID_RELATIONS = set(rel_type_map.keys())

    # A. 读取原有三元组 (保持骨架)
    print("Building Base Graph...")
    files = ['train.tsv', 'dev.tsv', 'test.tsv']
    for filename in files:
        path = KGE_DIR / filename
        if not path.exists(): continue
        df = pd.read_csv(path, sep='\t', header=None, names=['h', 'r', 't'])
        for _, row in df.iterrows():
            h, r, t = row['h'], row['r'], row['t']
            if r not in VALID_RELATIONS: continue
            if h not in ent2id or t not in ent2id: continue
            
            h_idx, t_idx = ent2id[h], ent2id[t]
            r_id = rel_type_map[r]
            
            # 双向
            edges_src.extend([h_idx, t_idx])
            edges_dst.extend([t_idx, h_idx])
            edges_type.extend([r_id, r_id + 5])
            
            if r == 'treats_disease':
                disease_herb_dict[t_idx].add(h_idx)
                all_herbs.add(h_idx)

    # B. 构建 BERT 语义边
    print("Building Semantic Edges (BERT)...")
    
    # Herb-Herb
    hh_src, hh_dst = build_edges_from_embeddings(herb_embs, herb_global_ids, SIM_THRESHOLD, TOP_K)
    print(f"   -> Added {len(hh_src)} Herb-Herb semantic edges.")
    edges_src.extend(hh_src)
    edges_dst.extend(hh_dst)
    edges_type.extend([REL_HERB_SEM] * len(hh_src))
    
    # Disease-Disease
    dd_src, dd_dst = build_edges_from_embeddings(disease_embs, disease_global_ids, SIM_THRESHOLD, TOP_K)
    print(f"   -> Added {len(dd_src)} Disease-Disease semantic edges.")
    edges_src.extend(dd_src)
    edges_dst.extend(dd_dst)
    edges_type.extend([REL_DISEASE_SEM] * len(dd_src))

    # 6. 保存
    print(f"Saving to {OUTPUT_DIR}...")
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
    
    torch.save(edge_index, OUTPUT_DIR / 'edge_index.pt')
    torch.save(edge_type, OUTPUT_DIR / 'edge_type.pt')
    torch.save(data_dict, OUTPUT_DIR / 'rec_data.pt')
    print("Done!")

if __name__ == "__main__":
    main()