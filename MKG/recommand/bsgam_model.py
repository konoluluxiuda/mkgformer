import torch
import torch.nn as nn
from torch_geometric.nn import MessagePassing


class GCN(MessagePassing):
    def __init__(self, in_channels, out_channels):
        super().__init__(aggr='mean')
        self.lin = nn.Linear(in_channels, out_channels)
        self.tanh = nn.Tanh()

    def forward(self, x, edge_index):
        out = self.propagate(edge_index, x=x)
        return self.tanh(out)

    def message(self, x_j):
        return self.lin(x_j)


class MultiHeadAtt(nn.Module):
    def __init__(self, embed_size, head_num, dropout):
        super().__init__()
        self.embed_size = embed_size
        self.head_num = head_num
        self.head_dim = embed_size // head_num

        if self.head_dim * head_num != embed_size:
            raise ValueError("embed_size must be divisible by head_num")

        self.q_proj = nn.Linear(embed_size, embed_size)
        self.k_proj = nn.Linear(embed_size, embed_size)
        self.v_proj = nn.Linear(embed_size, embed_size)
        self.out_proj = nn.Linear(embed_size, embed_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key, value, scale=None):
        bsz = query.size(0)
        q_len = query.size(1)
        k_len = key.size(1)

        q = self.q_proj(query).view(bsz, q_len, self.head_num, self.head_dim).transpose(1, 2)
        k = self.k_proj(key).view(bsz, k_len, self.head_num, self.head_dim).transpose(1, 2)
        v = self.v_proj(value).view(bsz, k_len, self.head_num, self.head_dim).transpose(1, 2)

        attn_scores = torch.matmul(q, k.transpose(-2, -1))
        if scale is None:
            scale = self.head_dim ** 0.5
        attn_scores = attn_scores / scale
        attn = torch.softmax(attn_scores, dim=-1)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(bsz, q_len, self.embed_size)
        return self.out_proj(out)


class BSGAMAdapted(nn.Module):
    """BSGAM dynamic variant for NEWHERB ranking experiments."""

    def __init__(
        self,
        num_diseases,
        num_herbs,
        input_dim,
        embedding_dim=64,
        head_num=4,
        att_drop=0.0,
        kg_dim=27,
    ):
        super().__init__()
        self.num_diseases = num_diseases
        self.num_herbs = num_herbs
        self.sh_num = num_diseases + num_herbs

        self.SH_s_mlp = nn.Linear(input_dim, embedding_dim)
        self.SH_s_bn = nn.BatchNorm1d(embedding_dim)
        self.convSH_TostudyS_1 = GCN(embedding_dim, embedding_dim)
        self.convSH_TostudyS_2 = GCN(embedding_dim, embedding_dim)
        self.SH_line_s_1 = nn.Linear(embedding_dim, embedding_dim)
        self.SH_line_s_2 = nn.Linear(embedding_dim, 256)
        self.SH_bn_s_1 = nn.BatchNorm1d(embedding_dim)
        self.SH_bn_s_2 = nn.BatchNorm1d(256)

        self.SH_h_mlp = nn.Linear(input_dim, embedding_dim)
        self.SH_h_bn = nn.BatchNorm1d(embedding_dim)
        self.convSH_TostudyS_1_h = GCN(embedding_dim, embedding_dim)
        self.convSH_TostudyS_2_h = GCN(embedding_dim, embedding_dim)
        self.SH_line_h_1 = nn.Linear(embedding_dim, embedding_dim)
        self.SH_line_h_2 = nn.Linear(embedding_dim, 256)
        self.SH_bn_h_1 = nn.BatchNorm1d(embedding_dim)
        self.SH_bn_h_2 = nn.BatchNorm1d(256)

        self.SS_s_mlp = nn.Linear(input_dim, embedding_dim)
        self.SS_s_bn = nn.BatchNorm1d(embedding_dim)
        self.convSS_1 = GCN(embedding_dim, embedding_dim)
        self.convSS_2 = GCN(embedding_dim, embedding_dim)
        self.SS_line_1 = nn.Linear(embedding_dim, embedding_dim)
        self.SS_line_2 = nn.Linear(embedding_dim, 256)
        self.SS_bn_1 = nn.BatchNorm1d(embedding_dim)
        self.SS_bn_2 = nn.BatchNorm1d(256)

        self.HH_h_mlp = nn.Linear(input_dim, embedding_dim)
        self.HH_h_bn = nn.BatchNorm1d(embedding_dim)
        self.kg_HH_mlp = nn.Linear(kg_dim, embedding_dim)
        self.kg_HH_bn = nn.BatchNorm1d(embedding_dim)
        self.convHH_1 = GCN(embedding_dim, embedding_dim)
        self.convHH_2 = GCN(embedding_dim, embedding_dim)
        self.HH_line_1 = nn.Linear(embedding_dim, embedding_dim)
        self.HH_line_2 = nn.Linear(embedding_dim, 256)
        self.HH_bn_1 = nn.BatchNorm1d(embedding_dim)
        self.HH_bn_2 = nn.BatchNorm1d(256)

        self.es_bn_1 = nn.BatchNorm1d(256)
        self.eh_bn_1 = nn.BatchNorm1d(256)
        self.mlp = nn.Linear(256, 256)
        self.SI_bn = nn.BatchNorm1d(256)

        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()

        self.attention_s = MultiHeadAtt(256, head_num, att_drop)
        self.attention_h = MultiHeadAtt(256, head_num, att_drop)

    def _compute_branches(self, sh_tensor, s_tensor, h_tensor, edge_index_SH, edge_index_SS, edge_index_HH, kgOneHot):
        esh0 = self.tanh(self.SH_s_bn(self.SH_s_mlp(sh_tensor)))
        b1_sh = self.tanh(self.SH_bn_s_1(self.SH_line_s_1(esh0 + self.convSH_TostudyS_1(esh0, edge_index_SH))))
        b2_sh = self.tanh(self.SH_bn_s_2(self.SH_line_s_2(b1_sh + self.convSH_TostudyS_2(b1_sh, edge_index_SH))))

        esh02 = self.tanh(self.SH_h_bn(self.SH_h_mlp(sh_tensor)))
        b1_sh2 = self.tanh(self.SH_bn_h_1(self.SH_line_h_1(esh02 + self.convSH_TostudyS_1_h(esh02, edge_index_SH))))
        b2_sh2 = self.tanh(self.SH_bn_h_2(self.SH_line_h_2(b1_sh2 + self.convSH_TostudyS_2_h(b1_sh2, edge_index_SH))))

        es0 = self.tanh(self.SS_s_bn(self.SS_s_mlp(s_tensor)))
        r1_s = self.tanh(self.SS_bn_1(self.SS_line_1(es0 + self.convSS_1(es0, edge_index_SS))))
        r2_s = self.tanh(self.SS_bn_2(self.SS_line_2(r1_s + self.convSS_2(r1_s, edge_index_SS))))

        eh0 = self.tanh(self.HH_h_bn(self.HH_h_mlp(h_tensor)))
        kg_h = self.tanh(self.kg_HH_bn(self.kg_HH_mlp(kgOneHot)))
        eh0_kg = eh0 + kg_h
        r1_h = self.tanh(self.HH_bn_1(self.HH_line_1(eh0_kg + self.convHH_1(eh0_kg, edge_index_HH))))
        r2_h = self.tanh(self.HH_bn_2(self.HH_line_2(r1_h + self.convHH_2(r1_h, edge_index_HH))))
        return b2_sh, b2_sh2, r2_s, r2_h

    def get_embeddings(self, sh_tensor, s_tensor, h_tensor, edge_index_SH, edge_index_SS, edge_index_HH, kgOneHot):
        b2_sh, b2_sh2, r2_s, r2_h = self._compute_branches(
            sh_tensor, s_tensor, h_tensor, edge_index_SH, edge_index_SS, edge_index_HH, kgOneHot
        )

        d = self.num_diseases
        h = self.num_herbs

        query_s = (b2_sh[:d] + r2_s).view(d, 1, 256)
        key_s = torch.cat((b2_sh[:d].view(d, 1, 256), r2_s.view(d, 1, 256)), dim=1)
        value_s = key_s
        es = self.attention_s(query_s, key_s, value_s).view(d, 256)
        es = self.tanh(self.es_bn_1(es))

        query_h = (b2_sh2[d:] + r2_h).view(h, 1, 256)
        key_h = torch.cat((b2_sh2[d:].view(h, 1, 256), r2_h.view(h, 1, 256)), dim=1)
        value_h = key_h
        eh = self.attention_h(query_h, key_h, value_h).view(h, 256)
        eh = self.tanh(self.eh_bn_1(eh))
        return es, eh

    def forward(self, sh_tensor, s_tensor, h_tensor, edge_index_SH, edge_index_SS, edge_index_HH, prescription, kgOneHot):
        es, eh = self.get_embeddings(sh_tensor, s_tensor, h_tensor, edge_index_SH, edge_index_SS, edge_index_HH, kgOneHot)
        e_synd = torch.mm(prescription, es)
        pre_sum = prescription.sum(dim=1, keepdim=True)
        e_synd_norm = e_synd / (pre_sum + 1e-9)
        e_synd_norm = self.relu(self.SI_bn(self.mlp(e_synd_norm)))
        return torch.mm(e_synd_norm, eh.t())
