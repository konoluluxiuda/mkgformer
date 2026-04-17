import os
import numpy as np
import torch
from torch.utils.data import DataLoader

from config import Config
from utils import set_seed, Evaluator
from presrecrf_model import PresRecRFAdapted
from presrecrf_wrapper import PresRecRFWrapper


CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
MKG_DIR = os.path.dirname(CURRENT_DIR)
DATA_ROOT = os.path.join(MKG_DIR, 'dataset', 'NEWHERB')
KDHR_DATA_DIR = os.path.join(DATA_ROOT, 'kdhr_newherb')
REC_DATA_DIR = os.path.join(DATA_ROOT, 'paper_graph_data')

PRESRF_CKPT = os.path.join(CURRENT_DIR, 'checkpoints', 'presrecrf_best.pt')
os.makedirs(os.path.dirname(PRESRF_CKPT), exist_ok=True)

PRESRF_LR = 7e-4
PRESRF_WEIGHT_DECAY = 7e-4
PRESRF_BATCH = Config.batch_size
PRESRF_EPOCHS = Config.epochs
PRESRF_EMB_DIM = Config.input_dim


class _PairDataset(torch.utils.data.Dataset):
    def __init__(self, d_local_array, h_pos_local_array, num_herbs, d_pos_sets):
        self.d_local_array = d_local_array
        self.h_pos_local_array = h_pos_local_array
        self.num_herbs = num_herbs
        self.d_pos_sets = d_pos_sets

    def __getitem__(self, idx):
        d = self.d_local_array[idx]
        pos = self.h_pos_local_array[idx]
        while True:
            neg = np.random.randint(0, self.num_herbs)
            if neg not in self.d_pos_sets[d]:
                break
        return d, pos, neg

    def __len__(self):
        return len(self.d_local_array)


def load_presrecrf_data(device):
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

    attr_path = os.path.join(REC_DATA_DIR, 'node_attributes.pt')
    if not os.path.exists(attr_path):
        raise FileNotFoundError(
            f"未找到 {attr_path}。请先生成 recommendation_data/node_attributes.pt"
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
    attr_full = torch.load(attr_path).float()

    if dense_full.size(0) != eval_meta['num_nodes']:
        raise ValueError(
            f"node_chem_dense.pt 行数({dense_full.size(0)})与 num_nodes({eval_meta['num_nodes']})不一致"
        )
    if attr_full.size(0) != eval_meta['num_nodes']:
        raise ValueError(
            f"node_attributes.pt 行数({attr_full.size(0)})与 num_nodes({eval_meta['num_nodes']})不一致"
        )

    disease_tensor = torch.tensor(disease_ids, dtype=torch.long)
    herb_tensor = torch.tensor(herb_indices, dtype=torch.long)

    feature_data = {
        'sym_sem': dense_full[disease_tensor].to(device),
        'herb_sem': dense_full[herb_tensor].to(device),
        'sym_struct': attr_full[disease_tensor].to(device),
        'herb_struct': attr_full[herb_tensor].to(device),
    }

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

    return {
        'feature_data': feature_data,
        'eval_meta': eval_meta,
        'train_d_local': train_d_local,
        'train_h_pos_local': train_h_pos_local,
        'd_pos_sets': d_pos_sets,
        'num_diseases': num_diseases,
        'num_herbs': num_herbs,
        'sem_dim': feature_data['sym_sem'].size(1),
        'struct_dim': feature_data['sym_struct'].size(1),
    }


def main():
    set_seed(Config.seed)
    device = torch.device(Config.device)
    print(f"[PresRecRF on NEWHERB] device={device}")
    print("Loading PresRecRF data (same train/test split as recommendation_data)...")
    data = load_presrecrf_data(device)

    feature_data = data['feature_data']
    eval_meta = data['eval_meta']
    num_diseases = data['num_diseases']
    num_herbs = data['num_herbs']

    train_dataset = _PairDataset(data['train_d_local'], data['train_h_pos_local'], num_herbs, data['d_pos_sets'])
    train_loader = DataLoader(train_dataset, batch_size=PRESRF_BATCH, shuffle=True)
    print(f"BPR train pairs: {len(train_dataset)}")

    model = PresRecRFAdapted(
        embedding_dim=PRESRF_EMB_DIM,
        herb_cnt=num_herbs,
        drop_ratio=0.1,
        sem_dim=data['sem_dim'],
        struct_dim=data['struct_dim'],
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=PRESRF_LR, weight_decay=PRESRF_WEIGHT_DECAY)

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
        f"\nStart Training PresRecRF (Max Epochs: {PRESRF_EPOCHS}, "
        f"Patience: {Config.patience}, Eval every: {Config.eval_interval})"
    )

    for epoch in range(PRESRF_EPOCHS):
        model.train()
        train_loss = 0.0

        for d_local, h_pos_local, h_neg_local in train_loader:
            d_local = d_local.to(device).long()
            h_pos_local = h_pos_local.to(device).long()
            h_neg_local = h_neg_local.to(device).long()

            optimizer.zero_grad()

            symptom_oh = torch.zeros(d_local.size(0), num_diseases, device=device)
            symptom_oh.scatter_(1, d_local.unsqueeze(1), 1.0)

            judge_herb, _ = model(
                symptom_oh,
                feature_data['sym_sem'],
                feature_data['herb_sem'],
                feature_data['sym_struct'],
                feature_data['herb_struct'],
            )

            pos_scores = torch.gather(judge_herb, 1, h_pos_local.unsqueeze(1)).squeeze(1)
            neg_scores = torch.gather(judge_herb, 1, h_neg_local.unsqueeze(1)).squeeze(1)
            loss = -torch.mean(torch.nn.functional.logsigmoid(pos_scores - neg_scores))

            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        avg_loss = train_loss / len(train_loader)

        if (epoch + 1) % Config.eval_interval != 0:
            continue

        print(f"Epoch {epoch+1} | Loss: {avg_loss:.4f}")
        wrapper = PresRecRFWrapper(model, feature_data, eval_meta, device=device)
        
        # ==== 修改点：评估时做早停参考的是 val_dict ====
        results = evaluator.evaluate(wrapper, val_dict, herb_indices, dummy_edge, dummy_type)
        res_str = " | ".join([f"{k}: {v:.4f}" for k, v in results.items() if "F1" in k])
        print(f"   >> [Validation] Metrics: {res_str}")

        cur_f1 = results['F1@10']
        if cur_f1 > best_f1:
            best_f1 = cur_f1
            no_improve_cnt = 0
            torch.save(model.state_dict(), PRESRF_CKPT)
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
    print("Final PresRecRF (NEWHERB) Test Results (same protocol as train.py)")
    print("=" * 50)
    
    # ==== 最终测试在未见过的 test_dict 上进行 ====
    model.load_state_dict(torch.load(PRESRF_CKPT, map_location=device))
    wrapper = PresRecRFWrapper(model, feature_data, eval_meta, device=device)
    results = evaluator.evaluate(wrapper, test_dict, herb_indices, dummy_edge, dummy_type)

    print("PresRecRF (NEWHERB) Test Results:")
    for k in Config.top_k:
        pk = results.get(f'Precision@{k}', 0)
        rk = results.get(f'Recall@{k}', 0)
        fk = results.get(f'F1@{k}', 0)
        print(f"  P@{k}={pk:.4f}  R@{k}={rk:.4f}  F1@{k}={fk:.4f}")


if __name__ == '__main__':
    main()