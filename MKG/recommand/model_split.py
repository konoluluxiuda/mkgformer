import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from config import Config

# =================================================================
# 1. 定义 BSGAM 风格的组件
# =================================================================

class ResidualGCNBlock(nn.Module):
    """
    复现 BSGAM/KDHR 的残差块: Output = Tanh(Linear(Input + GCN(Input)))
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.gcn = GCNConv(in_channels, out_channels)
        self.lin = nn.Linear(out_channels, out_channels)
        self.bn = nn.BatchNorm1d(out_channels)
        
        # 如果输入输出维度不一致，需要投影 Shortcut
        if in_channels != out_channels:
            self.shortcut = nn.Linear(in_channels, out_channels)
        else:
            self.shortcut = nn.Identity()

    def forward(self, x, edge_index):
        h = self.gcn(x, edge_index)
        res = self.shortcut(x) + h
        out = self.lin(res)
        out = self.bn(out)
        out = torch.tanh(out)
        return out

class BSGAM_Attention(nn.Module):
    """
    复现 BSGAM 的 Multi-Head Attention Fusion
    """
    def __init__(self, hidden_dim, num_heads=4, dropout=0.3):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.att = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=num_heads, batch_first=True)
        self.fc = nn.Linear(hidden_dim, hidden_dim)
        self.bn = nn.BatchNorm1d(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.kv_proj = nn.Linear(hidden_dim * 2, hidden_dim)

    def forward(self, x_view1, x_view2):
        query = (x_view1 + x_view2).unsqueeze(1)
        cat_feat = torch.cat([x_view1, x_view2], dim=-1)
        kv_feat = self.kv_proj(cat_feat).unsqueeze(1)
        
        attn_out, _ = self.att(query, kv_feat, kv_feat)
        out = attn_out.squeeze(1)
        out = self.dropout(out)
        out = self.fc(out)
        out = self.bn(out)
        out = torch.tanh(out)
        return out

# =================================================================
# 2. 主模型 (与 train.py 严格对齐模块的拆分图版本)
# =================================================================

class MultiView_GNN(nn.Module):
    def __init__(self, num_nodes, num_relations=12, pretrained_features=None, attr_matrix=None, chem_matrix=None, disease_matrix=None, fusion_mode='add'):
        super().__init__()
        
        self.emb_dim = Config.input_dim   
        self.hidden_dim = Config.hidden_dim 
        
        # 1. 基础 Embedding
        self.embedding = nn.Embedding(num_nodes, self.emb_dim)
        nn.init.xavier_uniform_(self.embedding.weight)
        
        # 2. 多模态语义特征维度对齐 (同 HMC_GNN_SSL)
        self.use_attr = False
        if attr_matrix is not None:
            self.use_attr = True
            self.register_buffer('attr_matrix', attr_matrix)
            self.attr_align = nn.Linear(attr_matrix.size(1), self.emb_dim)

        self.use_chem = False
        if chem_matrix is not None:
            self.use_chem = True
            self.register_buffer('chem_matrix', chem_matrix)
            self.chem_align = nn.Linear(chem_matrix.size(1), self.emb_dim)
            
        self.use_disease = False
        if disease_matrix is not None:
            self.use_disease = True
            self.register_buffer('disease_matrix', disease_matrix)
            self.disease_align = nn.Linear(disease_matrix.size(1), self.emb_dim)

        self.fusion_mode = fusion_mode
        self.use_gated_fusion = self.fusion_mode == 'gated' and self.use_attr and self.use_chem
        if self.use_gated_fusion:
            self.gate_st = nn.Linear(self.emb_dim, 1)
            self.gate_attr = nn.Linear(self.emb_dim, 1)
            self.gate_chem = nn.Linear(self.emb_dim, 1)

        self.fusion_mlp = nn.Sequential(
            nn.Linear(self.emb_dim, self.emb_dim),
            nn.ReLU()
        )

        # 3. 三个分支的残差 GCN 传播
        self.sh_gcn1 = ResidualGCNBlock(self.emb_dim, self.hidden_dim)
        self.sh_gcn2 = ResidualGCNBlock(self.hidden_dim, self.hidden_dim)
        
        self.ss_gcn1 = ResidualGCNBlock(self.emb_dim, self.hidden_dim)
        self.ss_gcn2 = ResidualGCNBlock(self.hidden_dim, self.hidden_dim)
        
        self.hh_gcn1 = ResidualGCNBlock(self.emb_dim, self.hidden_dim)
        self.hh_gcn2 = ResidualGCNBlock(self.hidden_dim, self.hidden_dim)
        
        # 4. Attention Fusion 模块
        self.fusion_disease = BSGAM_Attention(self.hidden_dim)
        self.fusion_herb = BSGAM_Attention(self.hidden_dim)
        
        self.final_mlp = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.BatchNorm1d(self.hidden_dim),
            nn.Tanh()
        )

    def forward_encoder(self, graphs, perturbed=False):
        edge_sh = graphs['sh']
        edge_ss = graphs['ss']
        edge_hh = graphs['hh']
        
        x_st = self.embedding.weight
        
        # === A. 早期节点特征融合 (与 train.py 中 HMC_GNN 完全一样) ===
        x_fused = x_st
        x_se1 = None
        x_se2 = None

        if self.use_attr:
            x_se1 = self.attr_align(self.attr_matrix)
            if not self.use_gated_fusion:
                x_fused = x_fused + x_se1
                
        if self.use_chem:
            x_se2 = self.chem_align(self.chem_matrix)
            if not self.use_gated_fusion:
                x_fused = x_fused + x_se2

        if self.use_gated_fusion and x_se1 is not None and x_se2 is not None:
            w_st = self.gate_st(x_st)
            w_attr = self.gate_attr(x_se1)
            w_chem = self.gate_chem(x_se2)
            weights = F.softmax(torch.cat([w_st, w_attr, w_chem], dim=-1), dim=-1)
            x_fused = weights[:, 0:1]*x_st + weights[:, 1:2]*x_se1 + weights[:, 2:3]*x_se2

        if self.use_disease:
            x_se3 = F.relu(self.disease_align(self.disease_matrix))
            disease_mask = (torch.sum(self.disease_matrix, dim=1) != 0).unsqueeze(1).float()
            x_fused = x_fused + x_se3 * disease_mask
            
        x_input = self.fusion_mlp(x_fused)

        # Edge Dropout 对扰动网络一致生效
        if perturbed and self.training:
            mask_sh = torch.rand(edge_sh.size(1), device=edge_sh.device) > Config.edge_drop_rate
            edge_sh = edge_sh[:, mask_sh]

        # === B. 独立的子图传播 (核心的消融差异：这里断开了图层级的信息交互) ===
        x_sh = self.sh_gcn1(x_input, edge_sh)
        x_sh = self.sh_gcn2(x_sh, edge_sh)

        x_ss = self.ss_gcn1(x_input, edge_ss)
        x_ss = self.ss_gcn2(x_ss, edge_ss)

        x_hh = self.hh_gcn1(x_input, edge_hh)
        x_hh = self.hh_gcn2(x_hh, edge_hh)

        # === C. 顶层视角的单独注意力组合 ===
        x_disease_final = self.fusion_disease(x_sh, x_ss)
        x_herb_final = self.fusion_herb(x_sh, x_hh)
        
        x_final = x_disease_final + x_herb_final
        x_final = self.final_mlp(x_final)
        
        return x_final

    # ---------------- 补充的所有 SSL 辅助函数 ----------------
    def calc_ssl_loss(self, x1, x2, nodes):
        z1 = F.normalize(x1[nodes], dim=1)
        z2 = F.normalize(x2[nodes], dim=1)
        sim = torch.matmul(z1, z2.t()) / Config.ssl_temp
        lbl = torch.arange(len(nodes), device=nodes.device)
        return F.cross_entropy(sim, lbl)

    def calc_cross_modal_loss(self, x_gnn, herb_indices):
        if not self.use_chem: return torch.tensor(0.0, device=x_gnn.device)
        z_gnn = F.normalize(x_gnn[herb_indices], dim=1)
        chem_buf = self.chem_matrix
        if not isinstance(chem_buf, torch.Tensor):
            chem_buf = torch.tensor(chem_buf, dtype=torch.float32, device=x_gnn.device)
        raw_chem_feat = chem_buf[herb_indices]
        z_chem = F.normalize(self.chem_align(raw_chem_feat), dim=1)
        sim_matrix = torch.matmul(z_gnn, z_chem.t()) / Config.ssl_temp
        labels = torch.arange(herb_indices.size(0), device=herb_indices.device)
        return F.cross_entropy(sim_matrix, labels)

    def calc_property_chem_loss(self, herb_indices):
        if (not self.use_attr) or (not self.use_chem): return torch.tensor(0.0, device=Config.device)
        attr_buf = self.attr_matrix
        chem_buf = self.chem_matrix
        if (not isinstance(attr_buf, torch.Tensor)) or (not isinstance(chem_buf, torch.Tensor)):
            return torch.tensor(0.0, device=Config.device)
        z_attr = F.normalize(self.attr_align(attr_buf[herb_indices]), dim=1)
        z_chem = F.normalize(self.chem_align(chem_buf[herb_indices]), dim=1)
        sim_matrix = torch.matmul(z_attr, z_chem.t()) / Config.ssl_temp
        labels = torch.arange(herb_indices.size(0), device=herb_indices.device)
        return F.cross_entropy(sim_matrix, labels)
