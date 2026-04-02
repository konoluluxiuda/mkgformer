import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA
from config import Config
from model import HMC_GNN_SSL
from dataset import GraphDataManager
import torch.nn.functional as F

# ==========================================
# matplotlib 论文级别配置
# ==========================================
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['axes.linewidth'] = 1.0

def load_model_and_extract_embs(model_path, use_ssl, data_manager):
    """
    实例化模型并分别提取药材节点的归一化模态表示向量。
    """
    chem_matrix = torch.load(os.path.join(Config.DATA_ROOT, 'recommendation_data', 'node_chem_dense.pt'))
    fp_path = os.path.join(Config.DATA_ROOT, 'recommendation_data', 'node_chem_fingerprint.pt')
    if os.path.exists(fp_path):
        fp_feat = torch.load(fp_path)
        chem_matrix = torch.cat([chem_matrix, fp_feat], dim=1)
    
    attr_tensors = []
    base_attr = data_manager.load_attributes()
    if base_attr is not None: attr_tensors.append(base_attr)
    chem_attr = torch.load(os.path.join(Config.DATA_ROOT, 'recommendation_data', 'node_chem_multihot.pt'))
    if chem_attr is not None: attr_tensors.append(chem_attr)
    final_attr_matrix = torch.cat(attr_tensors, dim=1)
    
    # 实例化模型
    model = HMC_GNN_SSL(
        num_nodes=data_manager.num_nodes,
        num_relations=data_manager.num_relations,
        pretrained_features=None,
        attr_matrix=final_attr_matrix,
        chem_matrix=chem_matrix,
        disease_matrix=None,  # 简化，只专注于药材的三模态
        fusion_mode='gated' if use_ssl else 'add'
    )
            
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location='cpu'), strict=False)
        print(f"✅ Loaded weights from {model_path}")
    else:
        print(f"⚠️ Warning: Model {model_path} not found. Using untrained INIT stats as Baseline.")

    model.eval()
    with torch.no_grad():
        herb_idx = torch.tensor(data_manager.herb_indices, dtype=torch.long)
        
        struct_emb = model.embedding(herb_idx)
        
        if model.use_attr:
            attr_emb = model.attr_align(final_attr_matrix[herb_idx].float())
        else:
            attr_emb = struct_emb.clone()
            
        if model.use_chem:
            chem_emb = model.chem_align(chem_matrix[herb_idx].float())
        else:
            chem_emb = struct_emb.clone()
            
        # 必须归一化，因为对比学习 Loss (InfoNCE) 优化的是余弦相似度
        struct_proj = F.normalize(struct_emb, dim=-1)
        attr_proj = F.normalize(attr_emb, dim=-1)
        chem_proj = F.normalize(chem_emb, dim=-1)
            
    return struct_proj.numpy(), attr_proj.numpy(), chem_proj.numpy()

def plot_3d_latent_space():
    print("1. Loading Data...")
    Config.REC_DATA_DIR = os.path.join(Config.DATA_ROOT, 'paper_graph_data')
    data_manager = GraphDataManager()
    data_manager.load_data()  
    
    np.random.seed(42)
    sample_mask = np.random.choice(len(data_manager.herb_indices), size=250, replace=False)
    
    print("2. Extracting Embeddings for [w/o SSL] model...")
    s1, a1, c1 = load_model_and_extract_embs('MISSING.pt', use_ssl=False, data_manager=data_manager)
    s1, a1, c1 = s1[sample_mask], a1[sample_mask], c1[sample_mask]
    
    print("3. Extracting Embeddings for [Ours SSL] model...")
    s2, a2, c2 = load_model_and_extract_embs('/home/zry/workspace/mkgformer/MKG/recommand/checkpoints/best_model.pt', use_ssl=True, data_manager=data_manager)
    s2, a2, c2 = s2[sample_mask], a2[sample_mask], c2[sample_mask]
    
    print("4. Applying Global PCA reduction...")
    # 使用 PCA 而不是 t-SNE。PCA 能保留全局的余弦相似度/分离度，完美契合 InfoNCE 目标
    pca1 = PCA(n_components=3, random_state=42)
    X1 = np.vstack((s1, a1, c1))
    X1_3d = pca1.fit_transform(X1)
    
    pca2 = PCA(n_components=3, random_state=42)
    X2 = np.vstack((s2, a2, c2))
    X2_3d = pca2.fit_transform(X2)
    
    n_pts = len(sample_mask)
    
    s1_3d, a1_3d, c1_3d = X1_3d[0:n_pts], X1_3d[n_pts:2*n_pts], X1_3d[2*n_pts:]
    s2_3d, a2_3d, c2_3d = X2_3d[0:n_pts], X2_3d[n_pts:2*n_pts], X2_3d[2*n_pts:]

    print("5. Plotting...")
    fig = plt.figure(figsize=(14, 6))
    
    labels = ['Structure (GNN)', 'TCM Attributes', 'Chemical (ChemBERTa)']
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1'] 
    markers = ['o', '^', 's']
    
    ax1 = fig.add_subplot(121, projection='3d')
    ax1.scatter(s1_3d[:,0], s1_3d[:,1], s1_3d[:,2], c=colors[0], label=labels[0], alpha=0.9, s=25, marker=markers[0], edgecolors='white', linewidth=0.5)
    ax1.scatter(a1_3d[:,0], a1_3d[:,1], a1_3d[:,2], c=colors[1], label=labels[1], alpha=0.9, s=25, marker=markers[1], edgecolors='white', linewidth=0.5)
    ax1.scatter(c1_3d[:,0], c1_3d[:,1], c1_3d[:,2], c=colors[2], label=labels[2], alpha=0.9, s=25, marker=markers[2], edgecolors='white', linewidth=0.5)
    ax1.set_title('(a) Direct Concat. (w/o SSL Alignment)', fontsize=14, pad=10)
    ax1.view_init(elev=20, azim=45)
    ax1.set_xticks([]); ax1.set_yticks([]); ax1.set_zticks([])
    
    ax2 = fig.add_subplot(122, projection='3d')
    ax2.scatter(s2_3d[:,0], s2_3d[:,1], s2_3d[:,2], c=colors[0], label=labels[0], alpha=0.9, s=25, marker=markers[0], edgecolors='white', linewidth=0.5)
    ax2.scatter(a2_3d[:,0], a2_3d[:,1], a2_3d[:,2], c=colors[1], label=labels[1], alpha=0.9, s=25, marker=markers[1], edgecolors='white', linewidth=0.5)
    ax2.scatter(c2_3d[:,0], c2_3d[:,1], c2_3d[:,2], c=colors[2], label=labels[2], alpha=0.9, s=25, marker=markers[2], edgecolors='white', linewidth=0.5)
    ax2.set_title('(b) Ours (w/ Cross-modal InfoNCE)', fontsize=14, pad=10)
    ax2.view_init(elev=20, azim=45)
    ax2.set_xticks([]); ax2.set_yticks([]); ax2.set_zticks([])
    
    handles, legend_labels = ax1.get_legend_handles_labels()
    fig.legend(handles, legend_labels, loc='lower center', ncol=3, fontsize=12, bbox_to_anchor=(0.5, 0.05), frameon=False)
    
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.15)
    os.makedirs('fig', exist_ok=True)
    out_path = 'fig/Latent_Space_Blocks.pdf'
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.savefig('fig/Latent_Space_Blocks.png', dpi=300, bbox_inches='tight')
    print(f"\n✅ Plot finished! Saved to {os.path.abspath(out_path)}")

if __name__ == "__main__":
    plot_3d_latent_space()
