import torch
import torch.nn as nn


class PresRecRFWrapper(nn.Module):
    """Wrap PresRecRFAdapted into forward_encoder API used by utils.Evaluator."""

    def __init__(self, model, feature_data, eval_meta, device=None):
        super().__init__()
        self.model = model
        self.feature_data = feature_data
        self.eval_meta = eval_meta
        self.device = device or next(model.parameters()).device
        self._emb_dim = model.embedding_dim

    def forward_encoder(self, edge_index, edge_type, perturbed=False):
        sym_sem = self.feature_data['sym_sem'].to(self.device)
        herb_sem = self.feature_data['herb_sem'].to(self.device)
        sym_struct = self.feature_data['sym_struct'].to(self.device)
        herb_struct = self.feature_data['herb_struct'].to(self.device)

        es, eh = self.model.get_embeddings(sym_sem, herb_sem, sym_struct, herb_struct)

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
