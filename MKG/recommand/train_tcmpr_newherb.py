import os
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from config import Config
from utils import set_seed, Evaluator
from tcmpr_model_adapted import TCMPRAdapted
from tcmpr_wrapper import TCMPRWrapper


CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
MKG_DIR = os.path.dirname(CURRENT_DIR)
DATA_ROOT = os.path.join(MKG_DIR, 'dataset', 'NEWHERB')
KDHR_DATA_DIR = os.path.join(DATA_ROOT, 'kdhr_newherb')
REC_DATA_DIR = os.path.join(DATA_ROOT, 'paper_graph_data')

TCMPR_CKPT = os.path.join(CURRENT_DIR, 'checkpoints', 'tcmpr_best.pt')
os.makedirs(os.path.dirname(TCMPR_CKPT), exist_ok=True)

# Keep TCMPR-like architecture while aligning training protocol.
MAX_SYMPTOM_NUM = 10
FUSION = 'avg'
LAYER1 = 256
LAYER2 = 64
EMBED_DIM = 128
CONV_FILTERS = 64
KERNEL_SIZE = 2
DROPOUT = 0.1

TCMPR_LR = 1e-3
TCMPR_WEIGHT_DECAY = 1e-5
TCMPR_BATCH = Config.batch_size
TCMPR_EPOCHS = Config.epochs


class _DiseaseDataset(Dataset):
    def __init__(self, d_local_array, labels_array):
        self.d_local_array = d_local_array
        self.labels_array = labels_array

    def __len__(self):
        return len(self.d_local_array)

    def __getitem__(self, idx):
        return self.d_local_array[idx], self.labels_array[idx]


def _build_symptom_sequences(dense_disease, max_symptom_num=10):
    """Build TCMPR-like symptom sequence via disease similarity neighborhood.

    Sequence format per disease: [self, top-neighbors...].
    """
    eps = 1e-12
    x = dense_disease
    x_norm = x / (torch.norm(x, dim=1, keepdim=True) + eps)
    sim = torch.matmul(x_norm, x_norm.t())

    n = x.size(0)
    seq_index = torch.zeros((n, max_symptom_num), dtype=torch.long)
    for i in range(n):
        row = sim[i].clone()
        row[i] = -1e9
        k = max_symptom_num - 1
        if k > 0:
            top_idx = torch.topk(row, k=min(k, n - 1)).indices
            seq = [i] + top_idx.tolist()
        else:
            seq = [i]

        if len(seq) < max_symptom_num:
            seq.extend([i] * (max_symptom_num - len(seq)))

        seq_index[i] = torch.tensor(seq[:max_symptom_num], dtype=torch.long)

    return dense_disease[seq_index]  # [N, L, D]


def load_tcmpr_data(device):
    eval_meta_path = os.path.join(KDHR_DATA_DIR, 'eval_meta.pt')
    if not os.path.exists(eval_meta_path):
        raise FileNotFoundError(
            f"未找到 {eval_meta_path}。请先运行 python MKG/recommand/build_kdhr_data_from_newherb.py"
        )

    dense_path = os.path.join(REC_DATA_DIR, 'node_chem_dense.pt')
    if not os.path.exists(dense_path):
        raise FileNotFoundError(
            f"未找到 {dense_path}。请先生成 paper_graph_data/node_chem_dense.pt"
        )

    eval_meta = torch.load(eval_meta_path)
    num_diseases = eval_meta['num_diseases']
    num_herbs = eval_meta['num_herbs']
    disease_ids = eval_meta['disease_ids']
    herb_indices = eval_meta['herb_indices']
    train_dict = eval_meta['train_dict']
    global_to_d = eval_meta['global_to_kdhr_disease']
    global_to_h = eval_meta['global_to_kdhr_herb']

    dense_full = torch.load(dense_path).float()
    if dense_full.size(0) != eval_meta['num_nodes']:
        raise ValueError(
            f"node_chem_dense.pt 行数({dense_full.size(0)})与 num_nodes({eval_meta['num_nodes']})不一致"
        )

    disease_tensor = torch.tensor(disease_ids, dtype=torch.long)
    dense_disease = dense_full[disease_tensor]

    symptom_seq_all = _build_symptom_sequences(dense_disease, max_symptom_num=MAX_SYMPTOM_NUM)

    # One training sample per disease (same style as original TCMPR case-level multi-label training).
    d_local_list = []
    y_list = []
    for d_global, herbs in train_dict.items():
        if d_global not in global_to_d:
            continue
        d_local = global_to_d[d_global]
        y = torch.zeros(num_herbs, dtype=torch.float32)
        for h_global in herbs:
            if h_global in global_to_h:
                y[global_to_h[h_global]] = 1.0
        if torch.sum(y) > 0:
            d_local_list.append(d_local)
            y_list.append(y)

    train_d_local = np.asarray(d_local_list, dtype=np.int64)
    train_y = torch.stack(y_list, dim=0) if y_list else torch.zeros((0, num_herbs), dtype=torch.float32)

    return {
        'symptom_seq_all': symptom_seq_all.to(device),
        'eval_meta': eval_meta,
        'train_d_local': train_d_local,
        'train_y': train_y,
        'num_diseases': num_diseases,
        'num_herbs': num_herbs,
        'symptom_dim': symptom_seq_all.size(2),
    }


def main():
    set_seed(Config.seed)
    device = torch.device(Config.device)
    print(f"[TCMPR on NEWHERB] device={device}")
    print("Loading TCMPR data (same train/test split as recommendation_data)...")

    data = load_tcmpr_data(device)
    symptom_seq_all = data['symptom_seq_all']
    eval_meta = data['eval_meta']

    train_dataset = _DiseaseDataset(data['train_d_local'], data['train_y'])
    train_loader = DataLoader(train_dataset, batch_size=TCMPR_BATCH, shuffle=True)
    print(f"TCMPR train queries: {len(train_dataset)}")

    model = TCMPRAdapted(
        symptom_dim=data['symptom_dim'],
        herb_count=data['num_herbs'],
        max_symptom_num=MAX_SYMPTOM_NUM,
        conv_filters=CONV_FILTERS,
        kernel_size=KERNEL_SIZE,
        fusion=FUSION,
        layer1=LAYER1,
        layer2=LAYER2,
        embed_dim=EMBED_DIM,
        dropout=DROPOUT,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=TCMPR_LR, weight_decay=TCMPR_WEIGHT_DECAY)

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

    print(
        f"\nStart Training TCMPR (Max Epochs: {TCMPR_EPOCHS}, "
        f"Patience: {Config.patience}, Eval every: {Config.eval_interval})"
    )

    for epoch in range(TCMPR_EPOCHS):
        model.train()
        train_loss = 0.0

        for d_local, y in train_loader:
            d_local = d_local.to(device).long()
            y = y.to(device).float()

            optimizer.zero_grad()
            x = symptom_seq_all[d_local]
            logits = model(x)
            loss = model.bce_multilabel_loss(logits, y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        avg_loss = train_loss / max(1, len(train_loader))

        if (epoch + 1) % Config.eval_interval != 0:
            continue

        print(f"Epoch {epoch+1} | Loss: {avg_loss:.4f}")
        wrapper = TCMPRWrapper(model, symptom_seq_all, eval_meta, device=device)
        
        # ==== 修改点：评估时使用 val_dict ====
        results = evaluator.evaluate(wrapper, val_dict, herb_indices, dummy_edge, dummy_type)
        res_str = " | ".join([f"{k}: {v:.4f}" for k, v in results.items() if "F1" in k])
        print(f"   >> [Validation] Metrics: {res_str}")

        cur_f1 = results['F1@10']
        if cur_f1 > best_f1:
            best_f1 = cur_f1
            no_improve_cnt = 0
            torch.save(model.state_dict(), TCMPR_CKPT)
            print(f"   >> ⭐ New Best Model! F1@10: {best_f1:.4f}")
        else:
            no_improve_cnt += 1
            print(f"   >> No improvement. Counter: {no_improve_cnt}/{Config.patience}")
            if no_improve_cnt >= Config.patience:
                print(
                    f"\n[Early Stopping] Triggered after "
                    f"{no_improve_cnt * Config.eval_interval} epochs without improvement."
                )
                print(f"Training Finished. Best F1@10 (Validation): {best_f1:.4f}")
                break

    print("\n" + "=" * 50)
    print("Final TCMPR (NEWHERB) Test Results (same protocol as train.py)")
    print("=" * 50)

    # ==== 最后载入最佳模型，在独立的 test_dict 上进行一次过境最终测试 ====
    model.load_state_dict(torch.load(TCMPR_CKPT, map_location=device))
    wrapper = TCMPRWrapper(model, symptom_seq_all, eval_meta, device=device)
    results = evaluator.evaluate(wrapper, test_dict, herb_indices, dummy_edge, dummy_type)

    print("TCMPR (NEWHERB) Test Results:")
    for k in Config.top_k:
        pk = results.get(f'Precision@{k}', 0)
        rk = results.get(f'Recall@{k}', 0)
        fk = results.get(f'F1@{k}', 0)
        print(f"  P@{k}={pk:.4f}  R@{k}={rk:.4f}  F1@{k}={fk:.4f}")

if __name__ == '__main__':
    main()