# train_kdhr_newherb.py
# 使用 NEWHERB 数据训练 KDHR，并用与 HMC_GNN_SSL 相同的 test_dict 和 utils.Evaluator 做评估，得到对比结果。
#
# 公平对比流程:
#   1) python MKG/recommand/preprocess_kge.py          # 生成 recommendation_data
#   2) python MKG/recommand/build_kdhr_data_from_newherb.py  # 从 REC 导出 KDHR 数据与同一 test_dict
#   3) python MKG/recommand/train_kdhr_newherb.py      # 训练 KDHR 并输出与 utils.Evaluator 一致指标
#   4) python MKG/recommand/train.py                   # 训练 HMC_GNN_SSL，用同一 test_dict 评估
# 对比时看两边的 P@K / R@K / F1@K 即可。
import os
import sys
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

# 将项目根目录加入 path 以便导入 KDHR
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
MKG_DIR = os.path.dirname(CURRENT_DIR)
PROJECT_ROOT = os.path.dirname(MKG_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from KDHR.model import KDHR

from config import Config
from utils import set_seed, Evaluator


class _PairDataset(torch.utils.data.Dataset):
    def __init__(self, d_local_array, h_pos_local_array, num_herbs, d_pos_sets):
        self.d_local_array = d_local_array
        self.h_pos_local_array = h_pos_local_array
        self.num_herbs = num_herbs
        self.d_pos_sets = d_pos_sets

    def __getitem__(self, idx):
        d = self.d_local_array[idx]
        pos = self.h_pos_local_array[idx]
        # 在 CPU 侧用 numpy 采样，避免 GPU .item() 阻塞
        while True:
            neg = np.random.randint(0, self.num_herbs)
            if neg not in self.d_pos_sets[d]:
                break
        return d, pos, neg

    def __len__(self):
        return len(self.d_local_array)
from kdhr_wrapper import KDHRWrapper

DATA_ROOT = os.path.join(MKG_DIR, 'dataset', 'NEWHERB')
KDHR_DATA_DIR = os.path.join(DATA_ROOT, 'kdhr_newherb')
KDHR_CKPT = os.path.join(CURRENT_DIR, 'checkpoints', 'kdhr_best.pt')
os.makedirs(os.path.dirname(KDHR_CKPT), exist_ok=True)

# KDHR 超参（与 KDHR 原论文接近）
KDHR_LR = 3e-4
KDHR_WEIGHT_DECAY = 7e-3
KDHR_DROP = 0.0
KDHR_BATCH = Config.batch_size
KDHR_EPOCHS = Config.epochs   # 与 recommand Config.epochs 对齐，由早停决定实际轮数
KDHR_EMB_DIM = 64
KDHR_KG_DIM = 27
# 早停与 recommand/train.py 完全一致：按 F1@10 + Config.eval_interval / Config.patience


def load_kdhr_data():
    if not os.path.exists(os.path.join(KDHR_DATA_DIR, 'eval_meta.pt')):
        raise FileNotFoundError(
            f"未找到 {KDHR_DATA_DIR}，请先运行:\n"
            "  1) python MKG/recommand/preprocess_kge.py\n"
            "  2) python MKG/recommand/build_kdhr_data_from_newherb.py"
        )

    eval_meta = torch.load(os.path.join(KDHR_DATA_DIR, 'eval_meta.pt'))
    num_diseases = eval_meta['num_diseases']
    num_herbs = eval_meta['num_herbs']
    sh_num = num_diseases + num_herbs
    kg_dim = eval_meta.get('kg_dim', KDHR_KG_DIM)

    # 图结构
    sh_edge = np.load(os.path.join(KDHR_DATA_DIR, 'sh_graph.npy'))  # [2, E]
    sh_x = torch.arange(sh_num, dtype=torch.float32).unsqueeze(1)  # [sh_num, 1]
    sh_edge_index = torch.tensor(sh_edge, dtype=torch.long)  # [2, E]

    ss_edge = np.load(os.path.join(KDHR_DATA_DIR, 'ss_graph.npy'))
    ss_x = torch.arange(num_diseases, dtype=torch.float32).unsqueeze(1)
    ss_edge_index = torch.tensor(ss_edge, dtype=torch.long)

    hh_edge = np.load(os.path.join(KDHR_DATA_DIR, 'hh_graph.npy'))
    # KDHR 中 H-H 图节点对应 SH 图中的草药节点索引: num_diseases .. sh_num-1
    hh_x = torch.arange(num_diseases, sh_num, dtype=torch.float32).unsqueeze(1)  # [num_herbs, 1]
    hh_edge_index = torch.tensor(hh_edge, dtype=torch.long)  # [2, E] 局部 0..num_herbs-1

    kg_oneHot = np.load(os.path.join(KDHR_DATA_DIR, 'herb_kg_oneHot.npy'))
    kg_oneHot = torch.from_numpy(kg_oneHot).float()  # [num_herbs, kg_dim]

    # 构造 disease-herb 正样本对（用于 BPR 排序训练）
    train_dict = eval_meta['train_dict']
    global_to_d = eval_meta['global_to_kdhr_disease']
    global_to_h = eval_meta['global_to_kdhr_herb']

    d_local_list = []
    h_pos_local_list = []
    d_pos_sets = {d: set() for d in range(num_diseases)}

    for d_global, herbs in train_dict.items():
        if d_global not in global_to_d:
            continue
        d_local = global_to_d[d_global]
        for h_global in herbs:
            if h_global not in global_to_h:
                continue
            h_local = global_to_h[h_global]
            d_local_list.append(d_local)
            h_pos_local_list.append(h_local)
            d_pos_sets[d_local].add(h_local)

    train_d_local = np.asarray(d_local_list, dtype=np.int64)
    train_h_pos_local = np.asarray(h_pos_local_list, dtype=np.int64)

    graph_data = {
        'sh_x': sh_x,
        'sh_edge': sh_edge_index,
        'ss_x': ss_x,
        'ss_edge': ss_edge_index,
        'hh_x': hh_x,
        'hh_edge': hh_edge_index,
        'kg_oneHot': kg_oneHot,
    }

    return {
        'graph_data': graph_data,
        'eval_meta': eval_meta,
        'train_d_local': train_d_local,
        'train_h_pos_local': train_h_pos_local,
        'd_pos_sets': d_pos_sets,
        'num_diseases': num_diseases,
        'num_herbs': num_herbs,
        'sh_num': sh_num,
        'kg_dim': kg_oneHot.shape[1],
    }


def main():
    set_seed(Config.seed)
    device = torch.device(Config.device)
    print(f"[KDHR on NEWHERB] device={device}")
    print("Loading KDHR data (same train/test split as recommendation_data)...")
    data = load_kdhr_data()
    graph_data = data['graph_data']
    eval_meta = data['eval_meta']
    num_diseases = data['num_diseases']
    num_herbs = data['num_herbs']
    sh_num = data['sh_num']
    kg_dim = data['kg_dim']

    train_dataset = _PairDataset(data['train_d_local'], data['train_h_pos_local'], num_herbs, data['d_pos_sets'])
    train_loader = DataLoader(train_dataset, batch_size=KDHR_BATCH, shuffle=True)
    print(f"BPR train pairs: {len(train_dataset)}")

    # 图与 KG 移到 device
    for k in graph_data:
        if isinstance(graph_data[k], torch.Tensor):
            graph_data[k] = graph_data[k].to(device)

    model = KDHR(
        num_diseases, num_herbs, sh_num,
        KDHR_EMB_DIM, KDHR_BATCH, KDHR_DROP, kg_dim=kg_dim
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=KDHR_LR, weight_decay=KDHR_WEIGHT_DECAY)

    evaluator = Evaluator(k_list=Config.top_k)
    dummy_edge = torch.zeros(2, 0, dtype=torch.long, device=device)
    dummy_type = torch.zeros(0, dtype=torch.long, device=device)
    
    # -------------------------------------------------------------------------
    # ========================== 关键修改：对齐 train.py 的数据切分 ==========================
    test_dict_original = eval_meta['test_dict']
    herb_indices = eval_meta['herb_indices']
    
    import random
    val_dict = {}
    new_test_dict = {}
    all_test_users = list(test_dict_original.keys())
    
    # 强制排序后打乱以确保完全复现 train.py 的划分结果
    all_test_users.sort() 
    random.seed(Config.seed)
    random.shuffle(all_test_users)
    
    half_idx = len(all_test_users) // 2
    for u in all_test_users[:half_idx]:
        val_dict[u] = test_dict_original[u]
    for u in all_test_users[half_idx:]:
        new_test_dict[u] = test_dict_original[u]
        
    test_dict = new_test_dict # 重置 test_dict 为真正独立的测试集
    
    print(f"✅ Data Split completed -> Val users: {len(val_dict)}, Test users: {len(test_dict)}")
    # ========================================================================================
    # -------------------------------------------------------------------------

    best_f1 = 0.0
    no_improve_cnt = 0

    print(f"\nStart Training KDHR (Max Epochs: {KDHR_EPOCHS}, Patience: {Config.patience}, Eval every: {Config.eval_interval})")
    for epoch in range(KDHR_EPOCHS):
        model.train()
        train_loss = 0.0
        for d_local, h_pos_local, h_neg_local in train_loader:
            d_local = d_local.to(device).long()
            h_pos_local = h_pos_local.to(device).long()
            h_neg_local = h_neg_local.to(device).long()

            optimizer.zero_grad()

            es, eh = model.get_embeddings(
                graph_data['sh_x'], graph_data['sh_edge'],
                graph_data['ss_x'], graph_data['ss_edge'],
                graph_data['hh_x'], graph_data['hh_edge'],
                graph_data['kg_oneHot']
            )

            d_emb = es[d_local]
            pos_emb = eh[h_pos_local]
            neg_emb = eh[h_neg_local]
            pos_scores = (d_emb * pos_emb).sum(dim=1)
            neg_scores = (d_emb * neg_emb).sum(dim=1)
            loss = -torch.mean(torch.nn.functional.logsigmoid(pos_scores - neg_scores))

            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        avg_loss = train_loss / len(train_loader)

        # --- 使用 val_dict 进行评估与早停 ---
        if (epoch + 1) % Config.eval_interval != 0:
            continue

        print(f"Epoch {epoch+1} | Loss: {avg_loss:.4f}")
        wrapper = KDHRWrapper(model, graph_data, eval_meta, device=device)
        
        # ==== 修改点：评估时参考 val_dict ====
        results = evaluator.evaluate(wrapper, val_dict, herb_indices, dummy_edge, dummy_type)
        res_str = " | ".join([f"{k}: {v:.4f}" for k, v in results.items() if "F1" in k])
        print(f"   >> [Validation] Metrics: {res_str}")

        cur_f1 = results["F1@10"]
        if cur_f1 > best_f1:
            best_f1 = cur_f1
            no_improve_cnt = 0
            torch.save(model.state_dict(), KDHR_CKPT)
            print(f"   >> ⭐ New Best Model! F1@10: {best_f1:.4f}")
        else:
            no_improve_cnt += 1
            print(f"   >> No improvement. Counter: {no_improve_cnt}/{Config.patience}")
            if no_improve_cnt >= Config.patience:
                print(f"\n[Early Stopping] Triggered after {no_improve_cnt * Config.eval_interval} epochs without improvement.")
                print(f"Training Finished. Best F1@10 (Validation): {best_f1:.4f}")
                break

    # ==== 最终测试在独立的 test_dict 上进行 ====
    print("\n" + "=" * 50)
    print("Final KDHR (NEWHERB) Test Results (same protocol as train.py)")
    print("=" * 50)
    model.load_state_dict(torch.load(KDHR_CKPT, map_location=device))
    wrapper = KDHRWrapper(model, graph_data, eval_meta, device=device)
    results = evaluator.evaluate(wrapper, test_dict, herb_indices, dummy_edge, dummy_type)

    print("KDHR (NEWHERB) Test Results:")
    for k in Config.top_k:
        pk, rk, fk = results.get(f'Precision@{k}', 0), results.get(f'Recall@{k}', 0), results.get(f'F1@{k}', 0)
        print(f"  P@{k}={pk:.4f}  R@{k}={rk:.4f}  F1@{k}={fk:.4f}")
    print("\nCompare with HMC_GNN_SSL by running: python MKG/recommand/train.py")
    print("(Use same recommendation_data and same top_k for fair comparison.)")


if __name__ == "__main__":
    main()
