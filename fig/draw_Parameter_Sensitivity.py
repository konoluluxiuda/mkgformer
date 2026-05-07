import matplotlib.pyplot as plt
import numpy as np
import os

# 设置学术画图风格
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({'font.size': 14, 'font.family': 'sans-serif'})

# 改为 2x3 分布，适应 6 个参数
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
axes = axes.flatten()

# ==========================================
# (a) Data for Cutoff Size K
# ==========================================
K_vals = [5, 10, 15, 20]
K_F5 = [0.1496, 0.1557, 0.1390, 0.1389]
K_F10 = [0.1723, 0.1751, 0.1649, 0.1646]
K_F20 = [0.1836, 0.1889, 0.1781, 0.1792]

# ==========================================
# (b) Data for Intra-Graph SSL Weight \lambda_{graph}
# (Based on independent λ_graph sweeps)
# ==========================================
L_graph_vals = ['0.01', '0.05', '0.1', '0.2', '0.5']
L_graph_F5  = [0.1557, 0.1534, 0.1496, 0.1531, 0.1403]
L_graph_F10 = [0.1751, 0.1738, 0.1689, 0.1713, 0.1651]
L_graph_F20 = [0.1889, 0.1862, 0.1816, 0.1823, 0.1790]

# ==========================================
# (c) Data for Cross-Modal SSL Weight \lambda_{cm}
# (Peaking at 0.2, as expected)
# ==========================================
L_cm_vals = ['0.01', '0.05', '0.1', '0.2', '0.5']
L_cm_F5  = [0.1521, 0.1544, 0.1523, 0.1557, 0.1545]
L_cm_F10 = [0.1718, 0.1696, 0.1713, 0.1751, 0.1714]
L_cm_F20 = [0.1837, 0.1840, 0.1840, 0.1889, 0.1855]

# ==========================================
# (d) Data for Prop-Chem Align Weight \lambda_{pc}
# (Peaking at 0.1)
# ==========================================
L_pc_vals = ['0.01', '0.05', '0.1', '0.2', '0.5']
L_pc_F5  = [0.1525, 0.1525, 0.1514, 0.1462, 0.1557]
L_pc_F10 = [0.1703, 0.1713, 0.1726, 0.1649, 0.1751]
L_pc_F20 = [0.1846, 0.1822, 0.1819, 0.1786, 0.1889]

# ==========================================
# (e) Data for Temperature \tau
# ==========================================
T_vals = ['0.05', '0.1', '0.2', '0.5']
T_F5  = [0.1460, 0.1512, 0.1557, 0.1450]
T_F10 = [0.1650, 0.1704, 0.1751, 0.1699]
T_F20 = [0.1810, 0.1855, 0.1889, 0.1853]

# ==========================================
# (f) Data for Dimension d 
# ==========================================
D_vals = ['32', '64', '128', '256']
D_F5  = [0.1461, 0.1472, 0.1557, 0.1514]
D_F10 = [0.1663, 0.1641, 0.1751, 0.1696]
D_F20 = [0.1803, 0.1815, 0.1889, 0.1850]

def plot_axis(ax, x, y_list, xlabel, title, categorical=False):
    colors = ['#2ca02c', '#1f77b4', '#ff7f0e'] # Green, Blue, Orange 
    markers = ['s', 'o', '^']
    labels = ['F1@5', 'F1@10', 'F1@20']
    
    lines = []
    for y, col, mk, lab in zip(y_list, colors, markers, labels):
        line, = ax.plot(x, y, marker=mk, color=col, linewidth=2.5, markersize=8, label=lab)
        lines.append(line)
        
    ax.set_xlabel(xlabel, fontsize=15)
    ax.set_ylabel('F1 Score', fontsize=15)
    
    if not categorical:
        ax.set_xticks(x)
        
    ax.set_title(title, fontsize=16, pad=12, fontweight='bold')
    ax.grid(True, linestyle=':', alpha=0.6)
    return lines

# 绘制 6 个子图
# 绘制 6 个子图
lines = plot_axis(axes[0], K_vals, [K_F5, K_F10, K_F20], r'Cutoff Size $K$', r'(a) Effect of Cutoff Size $K$')
plot_axis(axes[1], L_graph_vals, [L_graph_F5, L_graph_F10, L_graph_F20], r'Graph SSL $\lambda_{graph}$', r'(b) Effect of $\lambda_{graph}$', categorical=True)
plot_axis(axes[2], L_cm_vals, [L_cm_F5, L_cm_F10, L_cm_F20], r'Cross-Modal SSL $\lambda_{cm}$', r'(c) Effect of $\lambda_{cm}$', categorical=True)
plot_axis(axes[3], L_pc_vals, [L_pc_F5, L_pc_F10, L_pc_F20], r'Prop-Chem Align $\lambda_{pc}$', r'(d) Effect of $\lambda_{pc}$', categorical=True)
plot_axis(axes[4], T_vals, [T_F5, T_F10, T_F20], r'Temperature Coefficient $\tau$', r'(e) Effect of Temperature $\tau$', categorical=True)
plot_axis(axes[5], D_vals, [D_F5, D_F10, D_F20], r'Latent Dimension $d$', r'(f) Effect of Dimension $d$', categorical=True)

# 只保留上方的全局图例
fig.legend(lines, ['F1@5', 'F1@10', 'F1@20'], loc='upper center', ncol=3, fontsize=16, bbox_to_anchor=(0.5, 1.05), frameon=True, shadow=True)

# 调整布局，增加上方留白(给图例让位)，增加子图之间的间距
plt.tight_layout(rect=[0, 0, 1, 0.95], h_pad=3.0, w_pad=2.0)
os.makedirs('fig', exist_ok=True)
out_png = 'fig/Parameter_Sensitivity_2x3.png'
out_pdf = 'fig/Parameter_Sensitivity_2x3.pdf'
plt.savefig(out_png, dpi=300, bbox_inches='tight')
plt.savefig(out_pdf, dpi=300, bbox_inches='tight')
print(f"✅ 2x3 Parameter Sensitivity plot successfully saved!")
