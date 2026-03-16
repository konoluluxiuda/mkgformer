# kdhr_wrapper.py
# 将 KDHR 模型包装成与 HMC_GNN_SSL 相同的评估接口：forward_encoder(edge_index, edge_type, perturbed=False) -> [num_nodes, 256]
# 保证与 utils.Evaluator 和同一 test_dict 一致
import os
import torch
import torch.nn as nn

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
MKG_DIR = os.path.dirname(CURRENT_DIR)
PROJECT_ROOT = os.path.dirname(MKG_DIR)


def _import_kdhr():
    import sys
    if PROJECT_ROOT not in sys.path:
        sys.path.insert(0, PROJECT_ROOT)
    from KDHR.model import KDHR
    return KDHR


class KDHRWrapper(nn.Module):
    """包装 KDHR，提供 forward_encoder(edge_index, edge_type, perturbed=False) 返回 [num_nodes, 256]。"""

    def __init__(self, kdhr_model, graph_data, eval_meta, device=None):
        super().__init__()
        self.kdhr = kdhr_model
        self.graph_data = graph_data  # dict: sh_x, sh_edge, ss_x, ss_edge, hh_x, hh_edge, kg_oneHot
        self.eval_meta = eval_meta
        self.device = device or next(kdhr_model.parameters()).device
        self._emb_dim = 256

    def forward_encoder(self, edge_index, edge_type, perturbed=False):
        # 忽略 edge_index/edge_type，用 KDHR 自带的图计算嵌入
        sh_x = self.graph_data['sh_x'].to(self.device)
        sh_edge = self.graph_data['sh_edge'].to(self.device)
        ss_x = self.graph_data['ss_x'].to(self.device)
        ss_edge = self.graph_data['ss_edge'].to(self.device)
        hh_x = self.graph_data['hh_x'].to(self.device)
        hh_edge = self.graph_data['hh_edge'].to(self.device)
        kg_oneHot = self.graph_data['kg_oneHot'].to(self.device)

        es, eh = self.kdhr.get_embeddings(sh_x, sh_edge, ss_x, ss_edge, hh_x, hh_edge, kg_oneHot)
        num_nodes = self.eval_meta['num_nodes']
        disease_ids = self.eval_meta['disease_ids']
        herb_indices = self.eval_meta['herb_indices']
        global_to_d = self.eval_meta['global_to_kdhr_disease']
        global_to_h = self.eval_meta['global_to_kdhr_herb']

        full_emb = torch.zeros(num_nodes, self._emb_dim, device=self.device, dtype=es.dtype)
        for gid in disease_ids:
            if gid in global_to_d:
                full_emb[gid] = es[global_to_d[gid]]
        for gid in herb_indices:
            if gid in global_to_h:
                full_emb[gid] = eh[global_to_h[gid]]

        return full_emb
