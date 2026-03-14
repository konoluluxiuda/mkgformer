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
        self.lin = nn.Linear(out_channels, out_channels) # 论文源码中的 Linear 变换
        self.bn = nn.BatchNorm1d(out_channels)
        
        # 如果输入输出维度不一致，需要投影 Shortcut
        if in_channels != out_channels:
            self.shortcut = nn.Linear(in_channels, out_channels)
        else:
            self.shortcut = nn.Identity()

    def forward(self, x, edge_index):
        # 1. GCN 聚合
        h = self.gcn(x, edge_index)
        
        # 2. 残差连接 (Input + GCN_Output)
        # 注意: 论文源码逻辑是 x_new = Tanh(BN(Linear(x_old + gcn_out)))
        # 这里我们做微调以适配 PyG
        res = self.shortcut(x) + h
        
        # 3. 变换与激活
        out = self.lin(res)
        out = self.bn(out)
        out = torch.tanh(out) # 使用 Tanh
        return out

class BSGAM_Attention(nn.Module):
    """
    复现 BSGAM 的 Multi-Head Attention Fusion
    Query = Sum(View1, View2)
    Key = Value = Concat(View1, View2)
    """
    def __init__(self, hidden_dim, num_heads=4, dropout=0.3):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        
        # MultiheadAttention
        # embed_dim 指的是 Query 的维度
        self.att = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=num_heads, batch_first=True)
        
        # 输出层
        self.fc = nn.Linear(hidden_dim, hidden_dim)
        self.bn = nn.BatchNorm1d(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        
        # 因为 Key/Value 是拼接的 (Dim*2)，我们需要投影回 Dim 才能放入 MultiheadAttention
        # 或者我们修改 Query 的维度。
        # BSGAM 源码逻辑: Q, K, V 都有各自的 Linear 投影。
        # PyTorch 的 MultiheadAttention 内部自带投影，但要求 Q,K,V 输入维度一致(或者通过 kdim/vdim 指定)。
        
        # 为了适配: 我们先将 Concat 后的 K,V 投影回 hidden_dim
        self.kv_proj = nn.Linear(hidden_dim * 2, hidden_dim)

    def forward(self, x_view1, x_view2):
        # x_view1: S-H 图特征 [N, Dim]
        # x_view2: S-S/H-H 图特征 [N, Dim]
        
        # 1. 构造 Query (Sum)
        query = (x_view1 + x_view2).unsqueeze(1) # [N, 1, Dim]
        
        # 2. 构造 Key/Value (Concat -> Project)
        # 原始论文是用 Concat 作为 K/V 的源，然后内部投影
        cat_feat = torch.cat([x_view1, x_view2], dim=-1) # [N, 2*Dim]
        kv_feat = self.kv_proj(cat_feat).unsqueeze(1)    # [N, 1, Dim]
        
        # 3. Attention
        # PyTorch Attention: forward(query, key, value)
        attn_out, _ = self.att(query, kv_feat, kv_feat) # [N, 1, Dim]
        
        # 4. Residual + Norm (Transformer Block standard)
        out = attn_out.squeeze(1)
        out = self.dropout(out)
        out = self.fc(out) # 论文最后的 FC
        # 论文里这里似乎没有 Residual 加回 Query，但加了 BN
        out = self.bn(out)
        out = torch.tanh(out)
        
        return out

# =================================================================
# 2. 主模型
# =================================================================

class MultiView_GNN(nn.Module):
    def __init__(self, num_nodes, attr_matrix=None):
        super().__init__()
        
        self.emb_dim = Config.input_dim   # 128
        self.hidden_dim = Config.hidden_dim # 128
        
        # 1. 基础 Embedding
        self.embedding = nn.Embedding(num_nodes, self.emb_dim)
        nn.init.xavier_uniform_(self.embedding.weight)
        
        # 2. 属性注入 (H-H分支专用)
        self.use_attr = False
        if attr_matrix is not None:
            self.use_attr = True
            self.register_buffer('attr_matrix', attr_matrix)
            self.attr_proj = nn.Linear(attr_matrix.size(1), 64)
            # 这里的输入维度变化通过 GCN 的 in_channels 适配
            self.hh_in_dim = self.emb_dim + 64
        else:
            self.hh_in_dim = self.emb_dim

        # 3. 三个分支的残差 GCN
        # S-H Branch (2 layers)
        self.sh_gcn1 = ResidualGCNBlock(self.emb_dim, self.hidden_dim)
        self.sh_gcn2 = ResidualGCNBlock(self.hidden_dim, self.hidden_dim)
        
        # S-S Branch (2 layers)
        self.ss_gcn1 = ResidualGCNBlock(self.emb_dim, self.hidden_dim)
        self.ss_gcn2 = ResidualGCNBlock(self.hidden_dim, self.hidden_dim)
        
        # H-H Branch (2 layers)
        self.hh_gcn1 = ResidualGCNBlock(self.hh_in_dim, self.hidden_dim)
        self.hh_gcn2 = ResidualGCNBlock(self.hidden_dim, self.hidden_dim)
        
        # 4. Attention Fusion 模块
        # 专门针对 Herb 和 Disease 分别融合
        self.fusion_disease = BSGAM_Attention(self.hidden_dim)
        self.fusion_herb = BSGAM_Attention(self.hidden_dim)
        
        # 5. 最终预测层 (MLP)
        # 融合后的特征维度是 hidden_dim
        # 如果需要进一步变换
        self.final_mlp = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.BatchNorm1d(self.hidden_dim),
            nn.Tanh()
        )

    def forward_encoder(self, graphs, perturbed=False):
        edge_sh = graphs['sh']
        edge_ss = graphs['ss']
        edge_hh = graphs['hh']
        
        x0 = self.embedding.weight
        
        # SSL 扰动
        if perturbed and self.training:
            mask = torch.rand(edge_sh.size(1), device=edge_sh.device) > Config.edge_drop_rate
            edge_sh = edge_sh[:, mask]

        # --- Branch 1: S-H (二部图) ---
        x_sh = self.sh_gcn1(x0, edge_sh)
        x_sh = self.sh_gcn2(x_sh, edge_sh)
        # BSGAM 论文采用了 Residual 累加所有层，这里 ResidualBlock 内部已经做了
        # 所以 x_sh 就是最终的 S-H 表示

        # --- Branch 2: S-S (疾病协作) ---
        x_ss = self.ss_gcn1(x0, edge_ss)
        x_ss = self.ss_gcn2(x_ss, edge_ss)

        # --- Branch 3: H-H (草药协作 + 属性) ---
        if self.use_attr:
            # 论文中是: eh0_kg = eh0 + kgOneHoth0 (Add)
            # 我们保持之前的 Concat 优势，但在 ResidualBlock 内部会转为 hidden_dim
            attr_emb = F.relu(self.attr_proj(self.attr_matrix))
            x_hh_in = torch.cat([x0, attr_emb], dim=-1)
        else:
            x_hh_in = x0
            
        x_hh = self.hh_gcn1(x_hh_in, edge_hh)
        x_hh = self.hh_gcn2(x_hh, edge_hh)

        # --- Fusion (Attention) ---
        # 论文逻辑：
        # Disease Embedding = Att(Query=SH+SS, Key=SH|SS, Val=SH|SS)
        # Herb Embedding    = Att(Query=SH+HH, Key=SH|HH, Val=SH|HH)
        
        # x_sh 包含了所有节点，但在 S-S 融合时我们只关注 Disease 节点的特征
        # 但因为是全图操作，我们对所有节点做同样的融合运算
        
        # 融合 Disease 视角 (SH + SS)
        x_disease_final = self.fusion_disease(x_sh, x_ss)
        
        # 融合 Herb 视角 (SH + HH)
        x_herb_final = self.fusion_herb(x_sh, x_hh)
        
        # --- 组合最终输出 ---
        # 对于 Disease 节点，取 x_disease_final
        # 对于 Herb 节点，取 x_herb_final
        # 但在矩阵中无法通过索引简单区分（除非我们有 mask）
        # 简单策略：相加。因为 x_ss 对于 Herb 节点是无效的(孤立)，x_hh 对于 Disease 是无效的
        # 或者更严谨地，根据 dataset 中的 indices 可以在 loss 计算时取不同的 embedding
        
        # 这里为了兼容 evaluator 接口 (返回一个矩阵)，我们简单相加
        # 更好的做法是：train.py 里取的时候，Disease 取 x_disease_final, Herb 取 x_herb_final
        # 鉴于我们无法在 forward 里知道哪些 ID 是 Herb 哪些是 Disease
        # 我们返回一个融合后的 (N, Dim) 矩阵
        # 假设：x_ss 在 Herb 节点上的值接近噪声/0，x_hh 在 Disease 上同理
        
        x_final = x_disease_final + x_herb_final
        
        x_final = self.final_mlp(x_final)
        
        return x_final

    def calc_ssl_loss(self, x1, x2, nodes):
        z1 = F.normalize(x1[nodes], dim=1)
        z2 = F.normalize(x2[nodes], dim=1)
        sim = torch.matmul(z1, z2.t()) / Config.ssl_temp
        lbl = torch.arange(len(nodes), device=nodes.device)
        return F.cross_entropy(sim, lbl)