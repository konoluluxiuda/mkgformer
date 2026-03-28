# model.py (PresRecRF Fusion Version)
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import RGCNConv
from config import Config

class HMC_GNN_SSL(nn.Module):
    def __init__(self, num_nodes, num_relations, pretrained_features=None, attr_matrix=None, chem_matrix=None, disease_matrix=None, fusion_mode='add'):
        super().__init__()
        
        self.emb_dim = Config.input_dim   # 128
        self.hidden_dim = Config.hidden_dim # 128
        
        # ========================================================
        # 1. 结构特征 (Structural Feature - ST)
        # ========================================================
        self.embedding = nn.Embedding(num_nodes, self.emb_dim)
        nn.init.xavier_uniform_(self.embedding.weight)
        
        # ========================================================
        # 2. 多模态语义特征维度对齐 (Semantic Feature - SE Alignment)
        # ========================================================
        self.use_attr = False
        if attr_matrix is not None:
            self.use_attr = True
            self.register_buffer('attr_matrix', attr_matrix)
            # 对齐到 emb_dim (128)
            self.attr_align = nn.Linear(attr_matrix.size(1), self.emb_dim)

        self.use_chem = False
        if chem_matrix is not None:
            self.use_chem = True
            self.register_buffer('chem_matrix', chem_matrix)
            # 对齐到 emb_dim (128)
            self.chem_align = nn.Linear(chem_matrix.size(1), self.emb_dim)
            
        self.use_disease = False
        if disease_matrix is not None:
            self.use_disease = True
            self.register_buffer('disease_matrix', disease_matrix)
            # 对齐到 emb_dim (128)
            self.disease_align = nn.Linear(disease_matrix.size(1), self.emb_dim)

        self.fusion_mode = fusion_mode
        self.use_gated_fusion = self.fusion_mode == 'gated' and self.use_attr and self.use_chem
        if self.use_gated_fusion:
            # 节点级门控：为 (structure, attr, chem) 三路特征分配动态权重
            self.fusion_gate = nn.Sequential(
                nn.Linear(self.emb_dim * 3, self.emb_dim),
                nn.ReLU(),
                nn.Linear(self.emb_dim, 3)
            )

        # self.use_attr_chem_attn = self.use_attr and self.use_chem
        # if self.use_attr_chem_attn:
        #     self.attr_chem_gate = nn.Sequential(
        #         nn.Linear(self.emb_dim * 2, self.emb_dim),
        #         nn.ReLU(),
        #         nn.Linear(self.emb_dim, 2)  # 输出2个logit，对应 (attr, chem)
        #     )
        # ========================================================
        # 3. PresRecRF 融合层 (Fusion MLP)
        # ========================================================
        # 公式: Relu(W * (ST + SE1 + SE2) + b)
        self.fusion_mlp = nn.Sequential(
            nn.Linear(self.emb_dim, self.emb_dim),
            nn.ReLU()
        )

        # ========================================================
        # 4. RGCN 传播层
        # ========================================================
        # 输入已经是融合并对齐好的 emb_dim (128)
        self.conv1 = RGCNConv(self.emb_dim, self.hidden_dim, num_relations)
        self.bn1 = nn.BatchNorm1d(self.hidden_dim)
        
        self.conv2 = RGCNConv(self.hidden_dim, self.hidden_dim, num_relations)
        self.bn2 = nn.BatchNorm1d(self.hidden_dim)
        
        self.dropout = nn.Dropout(Config.dropout)
        
        # Layer Aggregation Fusion
        self.layer_fusion = nn.Linear(self.hidden_dim * 2, self.hidden_dim)

    def forward_encoder(self, edge_index, edge_type, perturbed=False):
        # 1. 获取结构特征 (ST)
        x_st = self.embedding.weight # [N, 128]
        
        # 2. 维度对齐与按维度相加 (Element-wise Addition ⊕)
        x_fused = x_st
        
        x_se1 = None
        x_se2 = None

        if self.use_attr:
            # 投影到 128 维
            attr_buf = self.attr_matrix
            if isinstance(attr_buf, torch.Tensor):
                x_se1 = self.attr_align(attr_buf)
                x_fused = x_fused + x_se1 # ⊕
            
        if self.use_chem:
            # 投影到 128 维
            chem_buf = self.chem_matrix
            if isinstance(chem_buf, torch.Tensor):
                x_se2 = self.chem_align(chem_buf)
                x_fused = x_fused + x_se2 # ⊕

        if self.use_gated_fusion and x_se1 is not None and x_se2 is not None:
            fusion_in = torch.cat([x_st, x_se1, x_se2], dim=-1)
            fusion_logits = self.fusion_gate(fusion_in)
            fusion_w = F.softmax(fusion_logits, dim=-1)
            w_st = fusion_w[:, 0:1]
            w_attr = fusion_w[:, 1:2]
            w_chem = fusion_w[:, 2:3]
            x_fused = w_st * x_st + w_attr * x_se1 + w_chem * x_se2

        if self.use_disease:
            disease_buf = self.disease_matrix
            if isinstance(disease_buf, torch.Tensor):
                x_se3 = self.disease_align(disease_buf)
                x_fused = x_fused + x_se3
            
        # # 1. 结构特征
        # x_st = self.embedding.weight  # [N, emb_dim]

        # # 2. 先分别对齐各模态
        # x_attr_aligned = None
        # x_chem_aligned = None

        # if self.use_attr:
        #     x_attr_aligned = self.attr_align(self.attr_matrix)   # [N, emb_dim]

        # if self.use_chem:
        #     x_chem_aligned = self.chem_align(self.chem_matrix)   # [N, emb_dim]

        # # 3. 模态融合
        # if self.use_attr_chem_attn:
        #     # 两个模态都存在，用 attention/gating 来融合
        #     # 拼接后过一层 MLP -> 得到每个节点两个权重
        #     cat = torch.cat([x_attr_aligned, x_chem_aligned], dim=-1)  # [N, 2*emb_dim]
        #     logits = self.attr_chem_gate(cat)                          # [N, 2]
        #     weights = F.softmax(logits, dim=-1)                        # [N, 2]

        #     w_attr = weights[:, 0].unsqueeze(-1)   # [N, 1]
        #     w_chem = weights[:, 1].unsqueeze(-1)   # [N, 1]

        #     x_sem = w_attr * x_attr_aligned + w_chem * x_chem_aligned  # [N, emb_dim]
        #     x_fused = x_st + x_sem
        # else:
        #     # 只有一个模态或都没有，则退回原来的简单加和
        #     x_fused = x_st
        #     if x_attr_aligned is not None:
        #         x_fused = x_fused + x_attr_aligned
        #     if x_chem_aligned is not None:
        #         x_fused = x_fused + x_chem_aligned
        # 3. PresRecRF 非线性融合映射 (ReLU MLP)
        # 此时 x_input 就是论文中的 e_i
        x_input = self.fusion_mlp(x_fused)
        
        # ==================================
        # 下面是常规的图传播 (Graph Propagation)
        # ==================================
        if perturbed and self.training:
            mask = torch.rand(edge_index.size(1), device=edge_index.device) > Config.edge_drop_rate
            edge_index = edge_index[:, mask]
            edge_type = edge_type[mask]

        x1 = self.conv1(x_input, edge_index, edge_type)
        x1 = self.bn1(x1)
        x1 = F.elu(x1)
        x1 = self.dropout(x1)
        
        x2 = self.conv2(x1, edge_index, edge_type)
        x2 = self.bn2(x2)
        x2 = F.elu(x2)
        x2 = self.dropout(x2)
        
        x_concat = torch.cat([x1, x2], dim=-1)
        x_final = self.layer_fusion(x_concat)
        
        return x_final

    # calc_ssl_loss 保持不变
    def calc_ssl_loss(self, x_view1, x_view2, unique_nodes):
        z1 = F.normalize(x_view1[unique_nodes], dim=1)
        z2 = F.normalize(x_view2[unique_nodes], dim=1)
        sim_matrix = torch.matmul(z1, z2.t()) / Config.ssl_temp
        labels = torch.arange(unique_nodes.size(0), device=unique_nodes.device)
        return F.cross_entropy(sim_matrix, labels)
    
    def calc_cross_modal_loss(self, x_gnn, herb_indices):
        """
        跨模态对比学习损失 (Cross-Modal SSL Loss)
        功能：将 GNN 提取的图语义特征与原始化学成分模态特征进行对齐。
        """
        if not self.use_chem:
            return torch.tensor(0.0, device=x_gnn.device)

        # 1. 提取 GNN 视图下的特征并归一化
        # x_gnn 是 forward_encoder 输出的 [N, 128]
        z_gnn = F.normalize(x_gnn[herb_indices], dim=1)

        # 2. 提取原始化学模态特征，投影到 128 维并归一化
        # 复用 init 中的 chem_align 映射层，确保空间一致
        chem_buf = self.chem_matrix
        if not isinstance(chem_buf, torch.Tensor):
            return torch.tensor(0.0, device=x_gnn.device)

        raw_chem_feat = chem_buf[herb_indices]
        z_chem = F.normalize(self.chem_align(raw_chem_feat), dim=1)

        # 3. 计算 InfoNCE 对比损失
        # 计算两组特征的余弦相似度矩阵 [Batch_Size, Batch_Size]
        # 期望：同一个 herb 的图特征和化学特征相似度最高（对角线）
        sim_matrix = torch.matmul(z_gnn, z_chem.t()) / Config.ssl_temp
        
        # 4. 生成标签（对角线即为正样本对）
        labels = torch.arange(herb_indices.size(0), device=herb_indices.device)
        
        # 5. 使用交叉熵计算对比损失
        cm_loss = F.cross_entropy(sim_matrix, labels)
        
        return cm_loss

    def calc_property_chem_loss(self, herb_indices):
        """
        属性-化学跨模态对齐损失。
        目标：让同一 herb 的性味归经表征与化学表征在共享空间中更接近。
        """
        if (not self.use_attr) or (not self.use_chem):
            return torch.tensor(0.0, device=self.embedding.weight.device)

        attr_buf = self.attr_matrix
        chem_buf = self.chem_matrix
        if (not isinstance(attr_buf, torch.Tensor)) or (not isinstance(chem_buf, torch.Tensor)):
            return torch.tensor(0.0, device=self.embedding.weight.device)

        z_attr = F.normalize(self.attr_align(attr_buf[herb_indices]), dim=1)
        z_chem = F.normalize(self.chem_align(chem_buf[herb_indices]), dim=1)

        sim_matrix = torch.matmul(z_attr, z_chem.t()) / Config.ssl_temp
        labels = torch.arange(herb_indices.size(0), device=herb_indices.device)
        return F.cross_entropy(sim_matrix, labels)