import torch
import torch.nn as nn


class TCMPRWrapper(nn.Module):
    """Wrap TCMPRAdapted into forward_encoder API used by utils.Evaluator."""

    def __init__(self, model, symptom_seq_all, eval_meta, device=None):
        super().__init__()
        self.model = model
        self.symptom_seq_all = symptom_seq_all
        self.eval_meta = eval_meta
        self.device = device or next(model.parameters()).device
        self._emb_dim = model.herb_emb.size(1)

    def forward_encoder(self, edge_index, edge_type, perturbed=False):
        symptom_seq_all = self.symptom_seq_all.to(self.device)
        disease_emb, herb_emb = self.model.get_embeddings(symptom_seq_all)

        num_nodes = self.eval_meta['num_nodes']
        disease_ids = self.eval_meta['disease_ids']
        herb_indices = self.eval_meta['herb_indices']
        global_to_d = self.eval_meta['global_to_kdhr_disease']
        global_to_h = self.eval_meta['global_to_kdhr_herb']

        full_emb = torch.zeros(num_nodes, self._emb_dim, device=self.device, dtype=disease_emb.dtype)

        for gid in disease_ids:
            if gid in global_to_d:
                full_emb[gid] = disease_emb[global_to_d[gid]]

        for gid in herb_indices:
            if gid in global_to_h:
                full_emb[gid] = herb_emb[global_to_h[gid]]

        return full_emb
