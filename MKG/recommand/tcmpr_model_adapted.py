import torch
import torch.nn as nn
import torch.nn.functional as F


class TCMPRAdapted(nn.Module):
    """TCMPR-style model adapted to NEWHERB with PyTorch.

    Core idea preserved:
    - Symptom-sequence encoder with Conv1D + pooling.
    - MLP projection for patient representation.
    - Herb scoring from patient representation.
    """

    def __init__(
        self,
        symptom_dim,
        herb_count,
        max_symptom_num=10,
        conv_filters=64,
        kernel_size=2,
        fusion="avg",
        layer1=256,
        layer2=128,
        embed_dim=128,
        dropout=0.1,
    ):
        super().__init__()
        self.symptom_dim = symptom_dim
        self.herb_count = herb_count
        self.max_symptom_num = max_symptom_num
        self.fusion = fusion.lower()

        self.conv = nn.Conv1d(
            in_channels=symptom_dim,
            out_channels=conv_filters,
            kernel_size=kernel_size,
            stride=1,
            padding=0,
        )

        if self.fusion == "max":
            self.pool = nn.AdaptiveMaxPool1d(1)
        else:
            self.pool = nn.AdaptiveAvgPool1d(1)

        self.mlp = nn.Sequential(
            nn.Linear(conv_filters, layer1),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(layer1, layer2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(layer2, embed_dim),
        )

        # Herb representations for inner-product recommendation.
        self.herb_emb = nn.Parameter(torch.empty(herb_count, embed_dim))
        nn.init.xavier_uniform_(self.herb_emb)

    def encode_symptom_sequence(self, symptom_seq):
        """symptom_seq: [B, L, D] -> disease embedding [B, E]."""
        x = symptom_seq.transpose(1, 2)  # [B, D, L]
        x = self.conv(x)                 # [B, C, L']
        x = self.pool(x).squeeze(-1)     # [B, C]
        x = self.mlp(x)                  # [B, E]
        return x

    def forward(self, symptom_seq):
        disease_emb = self.encode_symptom_sequence(symptom_seq)
        logits = torch.matmul(disease_emb, self.herb_emb.t())
        return logits

    def get_embeddings(self, all_symptom_seq):
        """Return full disease/herb embeddings for evaluator wrapper.

        all_symptom_seq: [N_disease, L, D]
        """
        disease_emb = self.encode_symptom_sequence(all_symptom_seq)
        herb_emb = self.herb_emb
        return disease_emb, herb_emb

    @staticmethod
    def bce_multilabel_loss(logits, labels):
        return F.binary_cross_entropy_with_logits(logits, labels)
