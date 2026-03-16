import os
import numpy as np
import torch
from torch.utils.data import DataLoader

from config import Config
from utils import set_seed, Evaluator
from bsgam_model import BSGAMAdapted
from bsgam_wrapper import BSGAMWrapper


CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
MKG_DIR = os.path.dirname(CURRENT_DIR)
DATA_ROOT = os.path.join(MKG_DIR, 'dataset', 'NEWHERB')
KDHR_DATA_DIR = os.path.join(DATA_ROOT, 'kdhr_newherb')
REC_DATA_DIR = os.path.join(DATA_ROOT, 'recommendation_data')

BSGAM_CKPT = os.path.join(CURRENT_DIR, 'checkpoints', 'bsgam_best.pt')
os.makedirs(os.path.dirname(BSGAM_CKPT), exist_ok=True)

# BSGAM hyper-parameters aligned to Config for fair comparison
BSGAM_LR = 1e-3
BSGAM_WEIGHT_DECAY = 1e-5
BSGAM_BATCH = Config.batch_size
BSGAM_EPOCHS = Config.epochs
BSGAM_EMB_DIM = 64
BSGAM_HEAD_NUM = 4
BSGAM_ATT_DROP = 0.0
BSGAM_KG_DIM = 27


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


def load_bsgam_data(device):
    eval_meta_path = os.path.join(KDHR_DATA_DIR, 'eval_meta.pt')
    if not os.path.exists(eval_meta_path):
        raise FileNotFoundError(
            f"未找到 {eval_meta_path}。请先运行 python MKG/recommand/build_kdhr_data_from_newherb.py"
        )

    dense_path = os.path.join(REC_DATA_DIR, 'node_chem_dense.pt')
    if not os.path.exists(dense_path):
        raise FileNotFoundError(
            f"未找到 {dense_path}。请先生成 recommendation_data/node_chem_dense.pt"
        )

    eval_meta = torch.load(eval_meta_path)
    num_diseases = eval_meta['num_diseases']
    num_herbs = eval_meta['num_herbs']
    disease_ids = eval_meta['disease_ids']
    herb_indices = eval_meta['herb_indices']
    train_dict = eval_meta['train_dict']
    global_to_d = eval_meta['global_to_kdhr_disease']
    global_to_h = eval_meta['global_to_kdhr_herb']

    sh_edge = torch.tensor(np.load(os.path.join(KDHR_DATA_DIR, 'sh_graph.npy')), dtype=torch.long)
    ss_edge = torch.tensor(np.load(os.path.join(KDHR_DATA_DIR, 'ss_graph.npy')), dtype=torch.long)
    hh_edge = torch.tensor(np.load(os.path.join(KDHR_DATA_DIR, 'hh_graph.npy')), dtype=torch.long)

    kg_oneHot = torch.from_numpy(np.load(os.path.join(KDHR_DATA_DIR, 'herb_kg_oneHot.npy'))).float()
    if kg_oneHot.size(1) != BSGAM_KG_DIM:
        if kg_oneHot.size(1) > BSGAM_KG_DIM:
            kg_oneHot = kg_oneHot[:, :BSGAM_KG_DIM]
        else:
            pad = torch.zeros(kg_oneHot.size(0), BSGAM_KG_DIM - kg_oneHot.size(1), dtype=kg_oneHot.dtype)
            kg_oneHot = torch.cat([kg_oneHot, pad], dim=1)

    dense_full = torch.load(dense_path).float()
    if dense_full.size(0) != eval_meta['num_nodes']:
        raise ValueError(
            f"node_chem_dense.pt 行数({dense_full.size(0)})与 num_nodes({eval_meta['num_nodes']})不一致"
        )

    sh_global_ids = disease_ids + herb_indices
    sh_tensor = dense_full[torch.tensor(sh_global_ids, dtype=torch.long)]
    s_tensor = dense_full[torch.tensor(disease_ids, dtype=torch.long)]
    h_tensor = dense_full[torch.tensor(herb_indices, dtype=torch.long)]

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
        'sh_tensor': sh_tensor.to(device),
        's_tensor': s_tensor.to(device),
        'h_tensor': h_tensor.to(device),
        'sh_edge': sh_edge.to(device),
        'ss_edge': ss_edge.to(device),
        'hh_edge': hh_edge.to(device),
        'kg_oneHot': kg_oneHot.to(device),
    }

    return {
        'graph_data': graph_data,
        'eval_meta': eval_meta,
        'train_d_local': train_d_local,
        'train_h_pos_local': train_h_pos_local,
        'd_pos_sets': d_pos_sets,
        'num_diseases': num_diseases,
        'num_herbs': num_herbs,
        'input_dim': sh_tensor.size(1),
    }


def main():
    set_seed(Config.seed)
    device = torch.device(Config.device)
    print(f"[BSGAM on NEWHERB] device={device}")
    print("Loading BSGAM data (same train/test split as recommendation_data)...")
    data = load_bsgam_data(device)

    graph_data = data['graph_data']
    eval_meta = data['eval_meta']
    num_diseases = data['num_diseases']
    num_herbs = data['num_herbs']
    input_dim = data['input_dim']

    train_dataset = _PairDataset(data['train_d_local'], data['train_h_pos_local'], num_herbs, data['d_pos_sets'])
    train_loader = DataLoader(train_dataset, batch_size=BSGAM_BATCH, shuffle=True)
    print(f"BPR train pairs: {len(train_dataset)}")

    model = BSGAMAdapted(
        num_diseases=num_diseases,
        num_herbs=num_herbs,
        input_dim=input_dim,
        embedding_dim=BSGAM_EMB_DIM,
        head_num=BSGAM_HEAD_NUM,
        att_drop=BSGAM_ATT_DROP,
        kg_dim=BSGAM_KG_DIM,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=BSGAM_LR, weight_decay=BSGAM_WEIGHT_DECAY)

    evaluator = Evaluator(k_list=Config.top_k)
    dummy_edge = torch.zeros(2, 0, dtype=torch.long, device=device)
    dummy_type = torch.zeros(0, dtype=torch.long, device=device)
    test_dict = eval_meta['test_dict']
    herb_indices = eval_meta['herb_indices']

    best_f1 = 0.0
    no_improve_cnt = 0

    print(
        f"\nStart Training BSGAM (Max Epochs: {BSGAM_EPOCHS}, "
        f"Patience: {Config.patience}, Eval every: {Config.eval_interval})"
    )

    for epoch in range(BSGAM_EPOCHS):
        model.train()
        train_loss = 0.0

        for d_local, h_pos_local, h_neg_local in train_loader:
            d_local = d_local.to(device).long()
            h_pos_local = h_pos_local.to(device).long()
            h_neg_local = h_neg_local.to(device).long()

            optimizer.zero_grad()
            es, eh = model.get_embeddings(
                graph_data['sh_tensor'], graph_data['s_tensor'], graph_data['h_tensor'],
                graph_data['sh_edge'], graph_data['ss_edge'], graph_data['hh_edge'],
                graph_data['kg_oneHot'],
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

        if (epoch + 1) % Config.eval_interval != 0:
            continue

        print(f"Epoch {epoch+1} | Loss: {avg_loss:.4f}")
        wrapper = BSGAMWrapper(model, graph_data, eval_meta, device=device)
        results = evaluator.evaluate(wrapper, test_dict, herb_indices, dummy_edge, dummy_type)
        res_str = " | ".join([f"{k}: {v:.4f}" for k, v in results.items() if "F1" in k])
        print(f"   >> Test Metrics: {res_str}")

        cur_f1 = results['F1@10']
        if cur_f1 > best_f1:
            best_f1 = cur_f1
            no_improve_cnt = 0
            torch.save(model.state_dict(), BSGAM_CKPT)
            print(f"   >> New Best Model! F1@10: {best_f1:.4f}")
        else:
            no_improve_cnt += 1
            print(f"   >> No improvement. Counter: {no_improve_cnt}/{Config.patience}")
            if no_improve_cnt >= Config.patience:
                print(
                    f"\n[Early Stopping] Triggered after "
                    f"{no_improve_cnt * Config.eval_interval} epochs without improvement."
                )
                print(f"Training Finished. Best F1@10: {best_f1:.4f}")
                break

    print("\n" + "=" * 50)
    print("Final BSGAM (NEWHERB) Test Results (same protocol as train.py)")
    print("=" * 50)
    model.load_state_dict(torch.load(BSGAM_CKPT, map_location=device))
    wrapper = BSGAMWrapper(model, graph_data, eval_meta, device=device)
    results = evaluator.evaluate(wrapper, test_dict, herb_indices, dummy_edge, dummy_type)

    print("BSGAM (NEWHERB) Test Results:")
    for k in Config.top_k:
        pk = results.get(f'Precision@{k}', 0)
        rk = results.get(f'Recall@{k}', 0)
        fk = results.get(f'F1@{k}', 0)
        print(f"  P@{k}={pk:.4f}  R@{k}={rk:.4f}  F1@{k}={fk:.4f}")


if __name__ == '__main__':
    main()
