import torch
import torch.nn as nn


class BSGAMWrapper(nn.Module):
    """Wrap BSGAMAdapted into forward_encoder API used by utils.Evaluator."""

    def __init__(self, bsgam_model, graph_data, eval_meta, device=None):
        super().__init__()
        self.bsgam = bsgam_model
        self.graph_data = graph_data
        self.eval_meta = eval_meta
        self.device = device or next(bsgam_model.parameters()).device
        self._emb_dim = 256

    def forward_encoder(self, edge_index, edge_type, perturbed=False):
        sh_tensor = self.graph_data['sh_tensor'].to(self.device)
        s_tensor = self.graph_data['s_tensor'].to(self.device)
        h_tensor = self.graph_data['h_tensor'].to(self.device)
        sh_edge = self.graph_data['sh_edge'].to(self.device)
        ss_edge = self.graph_data['ss_edge'].to(self.device)
        hh_edge = self.graph_data['hh_edge'].to(self.device)
        kg_oneHot = self.graph_data['kg_oneHot'].to(self.device)

        es, eh = self.bsgam.get_embeddings(
            sh_tensor, s_tensor, h_tensor,
            sh_edge, ss_edge, hh_edge,
            kg_oneHot,
        )

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
