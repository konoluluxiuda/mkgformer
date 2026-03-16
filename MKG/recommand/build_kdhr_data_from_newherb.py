# build_kdhr_data_from_newherb.py
# 从 NEWHERB recommendation_data 生成 KDHR 所需数据，保证 train/test 与 rec_data.pt 一致
# 使用方式: 先运行 preprocess_kge.py 生成 recommendation_data，再运行本脚本
import os
import numpy as np
import torch
from collections import defaultdict

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
MKG_DIR = os.path.dirname(CURRENT_DIR)
DATA_ROOT = os.path.join(MKG_DIR, 'dataset', 'NEWHERB')
REC_DIR = os.path.join(DATA_ROOT, 'recommendation_data')
OUT_DIR = os.path.join(DATA_ROOT, 'kdhr_newherb')
KG_DIM = 27
os.makedirs(OUT_DIR, exist_ok=True)


def main():
    rec_path = os.path.join(REC_DIR, 'rec_data.pt')
    edge_index_path = os.path.join(REC_DIR, 'edge_index.pt')
    edge_type_path = os.path.join(REC_DIR, 'edge_type.pt')

    if not os.path.exists(rec_path):
        raise FileNotFoundError(f"请先运行 preprocess_kge.py 生成 {REC_DIR} 下的 rec_data.pt")

    rec_data = torch.load(rec_path)
    edge_index = torch.load(edge_index_path)
    edge_type = torch.load(edge_type_path)

    num_nodes = rec_data['num_nodes']
    herb_indices = list(rec_data['herb_indices'])
    train_dict = rec_data['train_dict']
    test_dict = rec_data['test_dict']

    disease_ids = sorted(set(train_dict.keys()) | set(test_dict.keys()))
    num_diseases = len(disease_ids)
    num_herbs = len(herb_indices)
    sh_num = num_diseases + num_herbs

    global_to_d = {g: i for i, g in enumerate(disease_ids)}
    global_to_h = {g: i for i, g in enumerate(herb_indices)}

    print(f"KDHR 数据: num_diseases={num_diseases}, num_herbs={num_herbs}, sh_num={sh_num}")

    # 1) S-H 边: treats_disease 对应 type 0 (herb->disease) 或 5 (disease->herb)
    sh_src, sh_dst = [], []
    for i in range(edge_index.size(1)):
        t = edge_type[i].item()
        if t not in (0, 5):
            continue
        u, v = edge_index[0, i].item(), edge_index[1, i].item()
        if t == 0:  # src=herb, dst=disease
            if u in global_to_h and v in global_to_d:
                sh_src.append(global_to_d[v])
                sh_dst.append(num_diseases + global_to_h[u])
        else:  # t==5: src=disease, dst=herb
            if u in global_to_d and v in global_to_h:
                sh_src.append(global_to_d[u])
                sh_dst.append(num_diseases + global_to_h[v])

    sh_edge = np.array([sh_src, sh_dst], dtype=np.int64)
    np.save(os.path.join(OUT_DIR, 'sh_graph.npy'), sh_edge)
    print(f"  S-H edges: {sh_edge.shape[1]}")

    # 2) S-S 边: type == 11
    ss_src, ss_dst = [], []
    for i in range(edge_index.size(1)):
        if edge_type[i].item() != 11:
            continue
        u, v = edge_index[0, i].item(), edge_index[1, i].item()
        if u in global_to_d and v in global_to_d:
            ss_src.append(global_to_d[u])
            ss_dst.append(global_to_d[v])
    if ss_src:
        ss_edge = np.array([ss_src, ss_dst], dtype=np.int64)
        np.save(os.path.join(OUT_DIR, 'ss_graph.npy'), ss_edge)
        print(f"  S-S edges: {ss_edge.shape[1]}")
    else:
        np.save(os.path.join(OUT_DIR, 'ss_graph.npy'), np.zeros((2, 0), dtype=np.int64))
        print("  S-S edges: 0 (empty)")

    # 3) H-H 边: type == 10, 局部索引 0..num_herbs-1
    hh_src, hh_dst = [], []
    for i in range(edge_index.size(1)):
        if edge_type[i].item() != 10:
            continue
        u, v = edge_index[0, i].item(), edge_index[1, i].item()
        if u in global_to_h and v in global_to_h:
            hh_src.append(global_to_h[u])
            hh_dst.append(global_to_h[v])
    if hh_src:
        hh_edge = np.array([hh_src, hh_dst], dtype=np.int64)
        np.save(os.path.join(OUT_DIR, 'hh_graph.npy'), hh_edge)
        print(f"  H-H edges: {hh_edge.shape[1]}")
    else:
        np.save(os.path.join(OUT_DIR, 'hh_graph.npy'), np.zeros((2, 0), dtype=np.int64))
        print("  H-H edges: 0 (empty)")

    # 4) 处方训练集: 与 train_dict 一一对应
    pS_list = []
    pH_list = []
    for d_global in disease_ids:
        if d_global not in train_dict or len(train_dict[d_global]) == 0:
            continue
        sid = np.zeros(num_diseases, dtype=np.float32)
        sid[global_to_d[d_global]] = 1.0
        hid = np.zeros(num_herbs, dtype=np.float32)
        for h_global in train_dict[d_global]:
            if h_global in global_to_h:
                hid[global_to_h[h_global]] = 1.0
        pS_list.append(sid)
        pH_list.append(hid)

    pS_array = np.stack(pS_list).astype(np.float32)
    pH_array = np.stack(pH_list).astype(np.float32)
    np.save(os.path.join(OUT_DIR, 'train_pS.npy'), pS_array)
    np.save(os.path.join(OUT_DIR, 'train_pH.npy'), pH_array)
    print(f"  Train prescriptions: {len(pS_list)}")

    # 5) KG 属性矩阵: 仅草药行, 对齐到 KG_DIM
    attr_path = os.path.join(REC_DIR, 'node_attributes.pt')
    if os.path.exists(attr_path):
        attr_full = torch.load(attr_path)
        herb_attrs = attr_full[herb_indices].numpy()
        if herb_attrs.shape[1] >= KG_DIM:
            kg_oneHot = herb_attrs[:, :KG_DIM].astype(np.float32)
        else:
            pad = np.zeros((num_herbs, KG_DIM - herb_attrs.shape[1]), dtype=np.float32)
            kg_oneHot = np.hstack([herb_attrs.astype(np.float32), pad])
        np.save(os.path.join(OUT_DIR, 'herb_kg_oneHot.npy'), kg_oneHot)
        print(f"  KG one-hot: {kg_oneHot.shape}")
    else:
        kg_oneHot = np.zeros((num_herbs, KG_DIM), dtype=np.float32)
        np.save(os.path.join(OUT_DIR, 'herb_kg_oneHot.npy'), kg_oneHot)
        print("  KG one-hot: zeros (node_attributes.pt not found)")

    # 6) 评估元数据: 与 MKG 完全一致的 test_dict / herb_indices
    eval_meta = {
        'test_dict': test_dict,
        'herb_indices': herb_indices,
        'train_dict': train_dict,
        'disease_ids': disease_ids,
        'global_to_kdhr_disease': global_to_d,
        'global_to_kdhr_herb': global_to_h,
        'num_nodes': num_nodes,
        'num_diseases': num_diseases,
        'num_herbs': num_herbs,
        'kg_dim': KG_DIM,
    }
    torch.save(eval_meta, os.path.join(OUT_DIR, 'eval_meta.pt'))
    print(f"  Saved eval_meta.pt (same test_dict/herb_indices as REC).")
    print("Done.")


if __name__ == "__main__":
    main()
