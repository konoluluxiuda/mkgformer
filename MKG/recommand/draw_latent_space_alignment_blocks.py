import matplotlib.pyplot as plt
import matplotlib.patches as patches

# 开启 LaTeX 风格字体
plt.rcParams.update({'font.size': 12, 'font.family': 'sans-serif'})

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

# 定义模态配色
color_topo = '#4DBBD5'   # 拓扑 (蓝)
color_prop = '#00A087'   # 属性 (绿)
color_chem = '#E64B35'   # 化学 (红)

# 绘制单个特征向量 (由多个小方块组成)
def draw_vector(ax, x, y, length, size, color, alpha=1.0):
    for i in range(length):
        # 增加一点空隙和边框，更像数据结构
        rect = patches.Rectangle((x + i*(size+0.05), y), size, size, 
                                 linewidth=1, edgecolor='white', facecolor=color, alpha=alpha)
        ax.add_patch(rect)

# 向量参数
v_len = 8      # 每个特征的维度模拟(小方块数量)
sq_size = 0.4  # 小方块边长
x_offset = 1.0

# =============== 状态1：Before Alignment (左图) ===============
# 按照模态分布在不同空间区域
# 拓扑区域
ax1.text(2.5, 9.5, "Topology Space", ha='center', fontsize=12, fontweight='bold', color=color_topo)
draw_vector(ax1, 1, 8.5, v_len, sq_size, color_topo) # Herb A
draw_vector(ax1, 1, 7.8, v_len, sq_size, color_topo) # Herb B
draw_vector(ax1, 1, 7.1, v_len, sq_size, color_topo) # Herb C
ax1.text(0.8, 8.7, "Herb A", ha='right', fontsize=10)
ax1.text(0.8, 8.0, "Herb B", ha='right', fontsize=10)
ax1.text(0.8, 7.3, "Herb C", ha='right', fontsize=10)
ax1.add_patch(patches.Rectangle((0, 6.8), 5.5, 2.5, fill=False, edgecolor='gray', linestyle='--', alpha=0.5, lw=2))

# 属性区域
ax1.text(8.5, 9.5, "Property Space", ha='center', fontsize=12, fontweight='bold', color=color_prop)
draw_vector(ax1, 7, 8.5, v_len, sq_size, color_prop) # Herb A
draw_vector(ax1, 7, 7.8, v_len, sq_size, color_prop) # Herb B
draw_vector(ax1, 7, 7.1, v_len, sq_size, color_prop) # Herb C
ax1.add_patch(patches.Rectangle((6, 6.8), 5.5, 2.5, fill=False, edgecolor='gray', linestyle='--', alpha=0.5, lw=2))

# 化学区域
ax1.text(5.5, 3.5, "Chemical Space", ha='center', fontsize=12, fontweight='bold', color=color_chem)
draw_vector(ax1, 4, 2.5, v_len, sq_size, color_chem) # Herb A
draw_vector(ax1, 4, 1.8, v_len, sq_size, color_chem) # Herb B
draw_vector(ax1, 4, 1.1, v_len, sq_size, color_chem) # Herb C
ax1.add_patch(patches.Rectangle((3, 0.8), 5.5, 2.5, fill=False, edgecolor='gray', linestyle='--', alpha=0.5, lw=2))

# 绘制拉近的箭头 (以 Herb A 为例)
# Topology <-> Chemical
ax1.annotate("", xy=(5.5, 3.0), xytext=(3.0, 8.5), 
             arrowprops=dict(arrowstyle="<|-|>", color=color_topo, linestyle="-.", linewidth=2))
ax1.text(3.5, 5.5, r"$\mathcal{L}_{CM}$ Pull", fontsize=14, fontweight='bold', color='black', rotation=-55)

# Property <-> Chemical
ax1.annotate("", xy=(6.5, 3.0), xytext=(8.0, 8.5), 
             arrowprops=dict(arrowstyle="<|-|>", color=color_prop, linestyle="-.", linewidth=2))
ax1.text(7.5, 5.5, r"$\mathcal{L}_{PC}$ Pull", fontsize=14, fontweight='bold', color='black', rotation=55)

ax1.set_title("(a) Before Alignment: Modality-Isolated Feature Tensors", pad=20, fontweight='bold')
ax1.set_xlim(-1, 12); ax1.set_ylim(0, 11)
ax1.axis('off')

# =============== 状态2：After Alignment (右图) ===============
# 按实体对齐，特征在隐空间中相互融合/拼接
start_x = 2.0

# Herb A 聚集区
ax2.text(start_x + v_len*sq_size*1.5, 9.2, "Herb A Representation", ha='center', fontsize=12, fontweight='bold')
draw_vector(ax2, start_x, 8.5, v_len, sq_size, color_topo)
draw_vector(ax2, start_x + v_len*sq_size + 0.3, 8.5, v_len, sq_size, color_prop)
draw_vector(ax2, start_x + 2*(v_len*sq_size + 0.3), 8.5, v_len, sq_size, color_chem)
ax2.add_patch(patches.Rectangle((start_x-0.2, 8.3), 3*(v_len*sq_size + 0.3), 0.8, fill=False, edgecolor='gray', linestyle='-'))

# Herb B 聚集区
ax2.text(start_x + v_len*sq_size*1.5, 6.7, "Herb B Representation", ha='center', fontsize=12, fontweight='bold')
draw_vector(ax2, start_x, 6.0, v_len, sq_size, color_topo)
draw_vector(ax2, start_x + v_len*sq_size + 0.3, 6.0, v_len, sq_size, color_prop)
draw_vector(ax2, start_x + 2*(v_len*sq_size + 0.3), 6.0, v_len, sq_size, color_chem)
ax2.add_patch(patches.Rectangle((start_x-0.2, 5.8), 3*(v_len*sq_size + 0.3), 0.8, fill=False, edgecolor='gray', linestyle='-'))

# Herb C 聚集区
ax2.text(start_x + v_len*sq_size*1.5, 4.2, "Herb C Representation", ha='center', fontsize=12, fontweight='bold')
draw_vector(ax2, start_x, 3.5, v_len, sq_size, color_topo)
draw_vector(ax2, start_x + v_len*sq_size + 0.3, 3.5, v_len, sq_size, color_prop)
draw_vector(ax2, start_x + 2*(v_len*sq_size + 0.3), 3.5, v_len, sq_size, color_chem)
ax2.add_patch(patches.Rectangle((start_x-0.2, 3.3), 3*(v_len*sq_size + 0.3), 0.8, fill=False, edgecolor='gray', linestyle='-'))

# 大箭头指向 Gated Fusion (示意拼接对齐后的流向)
ax2.annotate("", xy=(start_x + v_len*sq_size*1.5, 1.5), xytext=(start_x + v_len*sq_size*1.5, 2.5),
             arrowprops=dict(facecolor='black', shrink=0.05, width=4, headwidth=12))
ax2.add_patch(patches.Rectangle((start_x + v_len*sq_size*0.5, 0.5), v_len*sq_size*2, 0.8, fill=True, facecolor='lightgray', edgecolor='k'))
ax2.text(start_x + v_len*sq_size*1.5, 0.9, "Cross-Modal Gated Fusion", ha='center', va='center', fontsize=12, fontweight='bold')

ax2.set_title("(b) After Alignment: Entity-Centric Isomorphic Manifold", pad=20, fontweight='bold')
ax2.set_xlim(0, 14); ax2.set_ylim(0, 11)
ax2.axis('off')

# =============== 生成图例 ===============
handles = [
    patches.Patch(color=color_topo, label='Topology Embedding ($Z_{geo}$)'),
    patches.Patch(color=color_prop, label='TCM Property Embedding ($Z_{prop}$)'),
    patches.Patch(color=color_chem, label='Chemical Embedding ($Z_{chem}$)')
]
fig.legend(handles=handles, loc='lower center', ncol=3, bbox_to_anchor=(0.5, -0.05), frameon=False, fontsize=13)

plt.tight_layout()
plt.subplots_adjust(bottom=0.1)
plt.savefig("Latent_Space_Blocks.pdf", format='pdf', dpi=300, bbox_inches='tight')
print("Successfully saved Latent_Space_Blocks.pdf")
