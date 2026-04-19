import matplotlib.pyplot as plt
import numpy as np
import os

# 设置学术画图风格
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({'font.size': 13, 'font.family': 'sans-serif'})

fig, axes = plt.subplots(1, 4, figsize=(22, 4.5))

# ==========================================
# (a) Data for Cutoff Size K
# ==========================================
K_vals = [5, 10, 15, 20]
K_F5 = [0.1496, 0.1557, 0.1390, 0.1389]
K_F10 = [0.1723, 0.1751, 0.1649, 0.1646]
K_F20 = [0.1836, 0.1889, 0.1781, 0.1792]

# ==========================================
# (b) Data for Contrastive Weight \lambda
# ==========================================
L_vals = ['0.01', '0.05', '0.1', '0.2', '0.5']
L_F5 = [0.1424, 0.1506, 0.1557, 0.1504, 0.1475]
L_F10 = [0.1668, 0.1712, 0.1751, 0.1714, 0.1683]
L_F20 = [0.1815, 0.1861, 0.1889, 0.1856, 0.1784]

# ==========================================
# (c) Data for Temperature \tau (Empty for now)
# ==========================================
T_vals = ['0.05', '0.1', '0.2', '0.5', '1.0']
T_F5 = T_F10 = T_F20 = [np.nan] * 5

# ==========================================
# (d) Data for Dimension d (Empty for now)
# ==========================================
D_vals = ['32', '64', '128', '256']
D_F5 = D_F10 = D_F20 = [np.nan] * 4

def plot_axis(ax, x, y_list, xlabel, title, categorical=False):
    # Colors for @5, @10, @20
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
    
    # 填充空数据时的占位提示
    if np.isnan(y_list[0]).all():
        ax.text(0.5, 0.5, 'Waiting for Data...', horizontalalignment='center', verticalalignment='center', 
                transform=ax.transAxes, fontsize=16, color='gray', alpha=0.6)
        
    return lines

lines = plot_axis(axes[0], K_vals, [K_F5, K_F10, K_F20], 'Cutoff Size $K$', '(a) Effect of Cutoff Size $K$')
plot_axis(axes[1], L_vals, [L_F5, L_F10, L_F20], 'Contrastive Weight $\lambda$', '(b) Effect of Weight $\lambda$', categorical=True)
plot_axis(axes[2], T_vals, [T_F5, T_F10, T_F20], r'Temperature Coefficient $\tau$', '(c) Effect of Temperature $\tau$', categorical=True)
plot_axis(axes[3], D_vals, [D_F5, D_F10, D_F20], 'Latent Dimension $d$', '(d) Effect of Dimension $d$', categorical=True)

# 统一放置一个图例在底部
fig.legend(lines, ['F1@5', 'F1@10', 'F1@20'], loc='lower center', ncol=3, fontsize=15, bbox_to_anchor=(0.5, -0.08), frameon=True, shadow=True)

plt.tight_layout()
os.makedirs('fig', exist_ok=True)
out_png = 'fig/Parameter_Sensitivity_1x4.png'
out_pdf = 'fig/Parameter_Sensitivity_1x4.pdf'
plt.savefig(out_png, dpi=300, bbox_inches='tight')
plt.savefig(out_pdf, dpi=300, bbox_inches='tight')
print(f"✅ Plot updated with F1@5, F1@10, and F1@20.")
