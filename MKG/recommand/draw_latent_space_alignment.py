import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patches as patches

# 开启 LaTeX 风格字体 (可选，提升学术感)
plt.rcParams.update({'font.size': 13, 'font.family': 'sans-serif'})

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# 定义图例映射
markers = {'Topology (GNN)': 's', 'TCM Property': '^', 'Chemical (MACCS)': 'o'}
colors = {'Herb A': '#E64B35', 'Herb B': '#4DBBD5', 'Herb C': '#00A087'} # 借鉴 SCI 常用色系 Nature 风格

# =============== 状态1：Before Alignment ===============
# 模态主导：所有拓扑特征聚在一起，所有化学特征聚在一起
pts_before = {
    'Herb A': {'Topology (GNN)': (2, 8), 'TCM Property': (8, 8), 'Chemical (MACCS)': (5, 2)},
    'Herb B': {'Topology (GNN)': (1.5, 7), 'TCM Property': (7.5, 7.5), 'Chemical (MACCS)': (4.5, 2.5)},
    'Herb C': {'Topology (GNN)': (2.5, 7.5), 'TCM Property': (8.5, 7), 'Chemical (MACCS)': (5.5, 1.5)}
}

for herb, mods in pts_before.items():
    for mod, coord in mods.items():
        ax1.scatter(coord[0], coord[1], marker=markers[mod], color=colors[herb], s=200, edgecolors='k', zorder=3)

# 绘制模态区域的背景椭圆 (示意语义鸿沟)
ellipse_t = patches.Ellipse((2, 7.5), 2.5, 2.5, angle=0, alpha=0.1, color='gray')
ellipse_p = patches.Ellipse((8, 7.5), 2.5, 2.5, angle=0, alpha=0.1, color='gray')
ellipse_c = patches.Ellipse((5, 2), 2.5, 2.5, angle=0, alpha=0.1, color='gray')
ax1.add_patch(ellipse_t)
ax1.add_patch(ellipse_p)
ax1.add_patch(ellipse_c)

ax1.text(2, 9, "Topology Subspace", ha='center', fontsize=11, fontstyle='italic')
ax1.text(8, 9, "Property Subspace", ha='center', fontsize=11, fontstyle='italic')
ax1.text(5, 0.5, "Chemical Subspace", ha='center', fontsize=11, fontstyle='italic')

# 画箭头示意 L_CM 和 L_PC 的拉力 (以 Herb A 为例)
a_t = pts_before['Herb A']['Topology (GNN)']
a_p = pts_before['Herb A']['TCM Property']
a_c = pts_before['Herb A']['Chemical (MACCS)']
# Topology <-> Chemical (L_CM)
ax1.annotate("", xy=a_c, xytext=a_t, arrowprops=dict(arrowstyle="<|-|>", color="gray", linestyle="--", linewidth=1.5))
ax1.text(3.3, 5, r"$\mathcal{L}_{CM}$", color="black", fontsize=14, weight='bold')
# Property <-> Chemical (L_PC)
ax1.annotate("", xy=a_c, xytext=a_p, arrowprops=dict(arrowstyle="<|-|>", color="gray", linestyle="--", linewidth=1.5))
ax1.text(6.8, 5, r"$\mathcal{L}_{PC}$", color="black", fontsize=14, weight='bold')

ax1.set_title("(a) Before Contrastive Alignment\n(Modality-dominated Space)", pad=15, weight='bold')
ax1.set_xlim(0, 10); ax1.set_ylim(0, 10)
ax1.set_xticks([]); ax1.set_yticks([]) # 隐藏坐标轴刻度

# =============== 状态2：After Alignment ===============
# 实体主导：同一个 Herb 的三种特征紧密聚簇
pts_after = {
    'Herb A': {'Topology (GNN)': (3, 7.2), 'TCM Property': (2.8, 6.8), 'Chemical (MACCS)': (3.2, 6.9)},
    'Herb B': {'Topology (GNN)': (7, 7), 'TCM Property': (7.2, 7.3), 'Chemical (MACCS)': (6.8, 7.1)},
    'Herb C': {'Topology (GNN)': (5, 3), 'TCM Property': (4.8, 2.7), 'Chemical (MACCS)': (5.2, 2.8)}
}

for herb, mods in pts_after.items():
    for mod, coord in mods.items():
        # 添加 label 用于生成图例 (仅第一次)
        label_str = mod if herb == 'Herb A' else ""
        ax2.scatter(coord[0], coord[1], marker=markers[mod], color=colors[herb], s=200, edgecolors='k', zorder=3, label=label_str)

# 绘制实体聚集的背景椭圆 (示意对齐后)
e_a = patches.Ellipse((3, 7), 1.5, 1.5, angle=0, alpha=0.15, color=colors['Herb A'])
e_b = patches.Ellipse((7, 7.1), 1.5, 1.5, angle=0, alpha=0.15, color=colors['Herb B'])
e_c = patches.Ellipse((5, 2.8), 1.5, 1.5, angle=0, alpha=0.15, color=colors['Herb C'])
ax2.add_patch(e_a); ax2.add_patch(e_b); ax2.add_patch(e_c)

ax2.text(3, 8, "Herb A Cluster", ha='center', fontsize=12, weight='bold', color=colors['Herb A'])
ax2.text(7, 8.2, "Herb B Cluster", ha='center', fontsize=12, weight='bold', color=colors['Herb B'])
ax2.text(5, 1.8, "Herb C Cluster", ha='center', fontsize=12, weight='bold', color=colors['Herb C'])

ax2.set_title("(b) After Contrastive Alignment\n(Entity-dominated Isomorphic Space)", pad=15, weight='bold')
ax2.set_xlim(0, 10); ax2.set_ylim(0, 10)
ax2.set_xticks([]); ax2.set_yticks([])

# =============== 生成图例 ===============
# 模态形状图例
handles_mode = [plt.Line2D([0], [0], marker=m, color='w', markerfacecolor='gray', markersize=12, markeredgecolor='k', label=l) for l, m in markers.items()]
# 实体颜色图例
handles_color = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=c, markersize=12, markeredgecolor='k', label=l) for l, c in colors.items()]

fig.legend(handles_mode + handles_color, 
           list(markers.keys()) + list(colors.keys()), 
           loc='lower center', ncol=6, bbox_to_anchor=(0.5, -0.05), frameon=False)

plt.tight_layout()
plt.subplots_adjust(bottom=0.15)
plt.savefig("Latent_Space_Alignment.pdf", format='pdf', dpi=300, bbox_inches='tight')
print("Successfully saved Latent_Space_Alignment.pdf")
