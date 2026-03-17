import torch
import torch.nn as nn


class PresRecRFAdapted(nn.Module):
    """PresRecRF adaptation for disease-herb ranking on NEWHERB."""

    def __init__(self, embedding_dim, herb_cnt, drop_ratio, sem_dim, struct_dim):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.herb_cnt = herb_cnt

        # Semantic projection
        self.mlp_sem_sym_1 = nn.Linear(sem_dim, 256)
        self.mlp_sem_sym_2 = nn.Linear(256, embedding_dim)
        self.mlp_sem_herb_1 = nn.Linear(sem_dim, 256)
        self.mlp_sem_herb_2 = nn.Linear(256, embedding_dim)

        # Structural projection
        self.mlp_struct_sym = nn.Linear(struct_dim, embedding_dim)
        self.mlp_struct_herb = nn.Linear(struct_dim, embedding_dim)

        # Symptom side MLP
        self.mlp_sym_1 = nn.Linear(embedding_dim, 256)
        self.mlp_sym_2 = nn.Linear(256, 256)
        self.mlp_sym_3 = nn.Linear(256, embedding_dim)

        # Herb side MLP
        self.mlp_herb_1 = nn.Linear(embedding_dim, 256)
        self.mlp_herb_2 = nn.Linear(256, 256)
        self.mlp_herb_3 = nn.Linear(256, embedding_dim)

        # Keep dosage head for architecture similarity (unused in BPR training)
        self.mlp_dosage = nn.Linear(herb_cnt, herb_cnt)

        self.dropout = nn.Dropout(drop_ratio)
        self.tanh = nn.Tanh()
        self.relu = nn.ReLU()

    def _build_tables(self, sym_sem, herb_sem, sym_struct, herb_struct):
        sem_sym = self.mlp_sem_sym_2(self.mlp_sem_sym_1(sym_sem))
        sem_sym = self.dropout(sem_sym)
        sem_herb = self.mlp_sem_herb_2(self.mlp_sem_herb_1(herb_sem))

        struct_sym = self.mlp_struct_sym(sym_struct)
        struct_herb = self.mlp_struct_herb(herb_struct)

        sym_base = sem_sym + struct_sym
        herb_base = sem_herb + struct_herb

        sym_emb = self.dropout(self.tanh(self.mlp_sym_1(sym_base)))
        sym_emb = self.dropout(self.tanh(self.mlp_sym_2(sym_emb)))
        sym_table = self.mlp_sym_3(sym_emb)

        herb_emb = self.tanh(self.mlp_herb_1(herb_base))
        herb_emb = self.tanh(self.mlp_herb_2(herb_emb))
        herb_table = self.mlp_herb_3(herb_emb)

        return sym_table, herb_table

    def get_embeddings(self, sym_sem, herb_sem, sym_struct, herb_struct):
        return self._build_tables(sym_sem, herb_sem, sym_struct, herb_struct)

    def forward(self, symptom_oh, sym_sem, herb_sem, sym_struct, herb_struct):
        sym_table, herb_table = self._build_tables(sym_sem, herb_sem, sym_struct, herb_struct)

        sym_agg = torch.mm(symptom_oh, sym_table)
        judge_herb = torch.mm(sym_agg, herb_table.t())

        # Returned for compatibility with original architecture shape.
        judge_dosage = self.mlp_dosage(self.relu(judge_herb))
        return judge_herb, judge_dosage
