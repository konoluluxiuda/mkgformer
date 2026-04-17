import os
import numpy as np
import torch

from config import Config
from utils import set_seed
from ptm_a_model import PTMASchemeA


CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
MKG_DIR = os.path.dirname(CURRENT_DIR)
DATA_ROOT = os.path.join(MKG_DIR, 'dataset', 'NEWHERB')
REC_DATA_DIR = os.path.join(DATA_ROOT, 'paper_graph_data')

PTM_K = 128
PTM_X = 4
PTM_ALPHA = 1.0
PTM_BETA = 0.1
PTM_BETA_BAR = 0.1
PTM_ETA = 1.0
PTM_ITERS = 800
PTM_LOG_INTERVAL = 1
PTM_CKPT = os.path.join(CURRENT_DIR, 'checkpoints', 'ptm_a_best_snapshot.npz')


def _load_rec_data():
    rec_path = os.path.join(REC_DATA_DIR, 'rec_data.pt')
    if not os.path.exists(rec_path):
        raise FileNotFoundError(
            f"未找到 {rec_path}。请先运行 python MKG/recommand/preprocess_paper_graph.py"
        )

    data = torch.load(rec_path)
    train_dict = data['train_dict']
    test_dict = data['test_dict']
    herb_indices = list(data['herb_indices'])

    # scheme-A: disease is used as single symptom token, so we need disease universe
    disease_ids = sorted(list(set(list(train_dict.keys()) + list(test_dict.keys()))))

    return train_dict, test_dict, herb_indices, disease_ids


def _evaluate_topk(model, test_dict, herb_indices, k_list):
    metrics = {k: {'p': [], 'r': [], 'f1': []} for k in k_list}

    for d_global, truth_list in test_dict.items():
        if len(truth_list) == 0:
            continue

        scores = model.score_disease(d_global)
        if scores.ndim != 1 or scores.shape[0] != len(herb_indices):
            continue

        max_k = max(k_list)
        top_local = np.argpartition(-scores, max_k - 1)[:max_k]
        top_local = top_local[np.argsort(-scores[top_local])]
        top_global = [herb_indices[i] for i in top_local]

        truth_set = set(truth_list)
        for k in k_list:
            rec_k = set(top_global[:k])
            hits = len(rec_k & truth_set)
            p = hits / k
            r = hits / len(truth_set)
            f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
            metrics[k]['p'].append(p)
            metrics[k]['r'].append(r)
            metrics[k]['f1'].append(f1)

    results = {}
    for k in k_list:
        results[f'Precision@{k}'] = float(np.mean(metrics[k]['p'])) if metrics[k]['p'] else 0.0
        results[f'Recall@{k}'] = float(np.mean(metrics[k]['r'])) if metrics[k]['r'] else 0.0
        results[f'F1@{k}'] = float(np.mean(metrics[k]['f1'])) if metrics[k]['f1'] else 0.0
    return results


def main():
    set_seed(Config.seed)
    os.makedirs(os.path.dirname(PTM_CKPT), exist_ok=True)

    print(f"[PTM-A Scheme-A on NEWHERB] seed={Config.seed}")
    print("Loading recommendation_data split...")
    train_dict, test_dict, herb_indices, disease_ids = _load_rec_data()

    # ========================== 关键修改：对齐 train.py 的数据切分 ==========================
    import random
    val_dict = {}
    new_test_dict = {}
    all_test_users = list(test_dict.keys())
    
    # 强制排序后打乱以确保完全复现 train.py 的划分结果
    all_test_users.sort() 
    random.seed(Config.seed)
    random.shuffle(all_test_users)
    
    half_idx = len(all_test_users) // 2
    for u in all_test_users[:half_idx]:
        val_dict[u] = test_dict[u]
    for u in all_test_users[half_idx:]:
        new_test_dict[u] = test_dict[u]
        
    test_dict = new_test_dict # 重置 test_dict 为真正独立的测试集
    # ========================================================================================

    print(
        f"✅ Data Split completed -> Train users: {len(train_dict)}, "
        f"Val users: {len(val_dict)}, Test users: {len(test_dict)}"
    )

    model = PTMASchemeA(
        num_topics=PTM_K,
        num_roles=PTM_X,
        alpha=PTM_ALPHA,
        beta=PTM_BETA,
        beta_bar=PTM_BETA_BAR,
        eta=PTM_ETA,
        iterations=PTM_ITERS,
        log_interval=PTM_LOG_INTERVAL,
        seed=Config.seed,
    )

    print(
        f"Training PTM-A: K={PTM_K}, X={PTM_X}, iters={PTM_ITERS}, "
        f"alpha={PTM_ALPHA}, beta={PTM_BETA}, beta_bar={PTM_BETA_BAR}, eta={PTM_ETA}"
    )
    print("[PTM-A] Gibbs sampling started. Iteration progress will be printed.")

    best_f1 = 0.0
    no_improve_cnt = 0
    best_snapshot = None

    print(
        f"Start PTM Early-Stopping: Eval every {Config.eval_interval} iters, "
        f"Patience: {Config.patience}"
    )

    def _on_iter_end(model_obj, iter_idx, total_iters):
        nonlocal best_f1, no_improve_cnt, best_snapshot

        if iter_idx % Config.eval_interval != 0:
            return False

        model_obj.refresh_posterior()
        
        # ==== 修改点：评估时使用 val_dict，而不是 test_dict ====
        results = _evaluate_topk(model_obj, val_dict, herb_indices, Config.top_k)
        
        res_str = " | ".join([f"{k}: {v:.4f}" for k, v in results.items() if 'F1' in k])
        print(f"Iter {iter_idx}/{total_iters} | [Validation] Metrics: {res_str}")

        cur_f1 = results['F1@10']
        if cur_f1 > best_f1:
            best_f1 = cur_f1
            no_improve_cnt = 0
            best_snapshot = model_obj.get_snapshot()
            print(f"   >> ⭐ New Best Model! F1@10: {best_f1:.4f}")
        else:
            no_improve_cnt += 1
            print(f"   >> No improvement. Counter: {no_improve_cnt}/{Config.patience}")
            if no_improve_cnt >= Config.patience:
                print(
                    f"[Early Stopping] Triggered after "
                    f"{no_improve_cnt * Config.eval_interval} iterations without improvement."
                )
                return True
        return False

    model.fit(
        train_dict=train_dict,
        herb_indices=herb_indices,
        disease_ids=disease_ids,
        max_iterations=PTM_ITERS,
        on_iteration_end=_on_iter_end,
    )

    if best_snapshot is not None:
        model.load_snapshot(best_snapshot)
        np.savez(
            PTM_CKPT,
            theta=best_snapshot['theta'],
            phi=best_snapshot['phi'],
            phi_bar=best_snapshot['phi_bar'],
            pi=best_snapshot['pi'],
        )
        print(f"Best PTM snapshot saved: {PTM_CKPT}")
        print(f"Validation Finished. Best F1@10: {best_f1:.4f}")
    else:
        model.refresh_posterior()
        print("Warning: No eval step was triggered; using final iteration parameters.")

    # ==== 修改点：最终测试在独立划分出的 new_test_dict 上进行 ====
    print("\nEvaluating with same Top-K protocol on final TEST set...")
    results = _evaluate_topk(model, test_dict, herb_indices, Config.top_k)

    print("\n" + "=" * 50)
    print("Final PTM-A (Scheme-A) Test Results (Same Protocol as HMC-GNN)")
    print("=" * 50)
    for k in Config.top_k:
        pk = results.get(f'Precision@{k}', 0.0)
        rk = results.get(f'Recall@{k}', 0.0)
        fk = results.get(f'F1@{k}', 0.0)
        print(f"  P@{k}={pk:.4f}  R@{k}={rk:.4f}  F1@{k}={fk:.4f}")

if __name__ == '__main__':
    main()