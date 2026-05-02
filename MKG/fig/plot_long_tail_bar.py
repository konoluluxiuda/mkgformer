import matplotlib.pyplot as plt
import numpy as np

# 设置绘图风格
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.sans-serif'] = ['Arial'] # 确保能显示英文字体
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.size'] = 14

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

labels = ['Head (Top-20%)', 'Mid (Mid-40%)', 'Tail (Btm-40%)']
x = np.arange(len(labels))
width = 0.25

# ==========================================
# 1. Baseline 柱状图对比
# ==========================================
# 数据
bsgam = [0.4901, 0.0095, 0.0000]
kdhr = [0.4058, 0.1974, 0.1048]
hmc = [0.4309, 0.2449, 0.1026]

# 绘制柱子，赋予不同颜色，使用更科学学术的配色
rects1 = axes[0].bar(x - width, bsgam, width, label='BSGAM', color='#E69F00', edgecolor='k')
rects2 = axes[0].bar(x, kdhr, width, label='KDHR', color='#56B4E9', edgecolor='k')
rects3 = axes[0].bar(x + width, hmc, width, label='HMC-GNN (Ours)', color='#009E73', edgecolor='k')

# 标注数值
def autolabel(rects, ax):
    for rect in rects:
        height = rect.get_height()
        ax.annotate(f'{height:.4f}',
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=10, rotation=90)

autolabel(rects1, axes[0])
autolabel(rects2, axes[0])
autolabel(rects3, axes[0])

axes[0].set_ylabel('Recall@10', fontweight='bold')
axes[0].set_title('(a) Long-tail Mitigation vs. Baselines', fontweight='bold', pad=20)
axes[0].set_xticks(x)
axes[0].set_xticklabels(labels, fontweight='bold')
axes[0].legend(loc='upper right')
axes[0].set_ylim(0, 0.6) # 统一纵坐标限制，留出放标签的空间

# ==========================================
# 2. Ablation 柱状图对比 (图构建策略消融)
# ==========================================
# 数据：连很多边 (Dense Graph), 不连边 (No Edge), HMC-GNN (Top-K)
dense = [0.5090, 0.1831, 0.0724]
noedge = [0.4916, 0.2076, 0.0744]
hmc_ablation = [0.4309, 0.2449, 0.1026] # 也就是 HMC，作为对照组应该放在最后或中间

rects4 = axes[1].bar(x - width, dense, width, label='Dense Graph (No Pruning)', color='#D55E00', edgecolor='k')
rects5 = axes[1].bar(x, noedge, width, label='No Edge (No Collaboration)', color='#CC79A7', edgecolor='k')
rects6 = axes[1].bar(x + width, hmc_ablation, width, label='HMC-GNN (Top-K Jaccard)', color='#009E73', edgecolor='k')

autolabel(rects4, axes[1])
autolabel(rects5, axes[1])
autolabel(rects6, axes[1])

axes[1].set_ylabel('Recall@10', fontweight='bold')
axes[1].set_title('(b) Graph Construction Strategy Ablation', fontweight='bold', pad=20)
axes[1].set_xticks(x)
axes[1].set_xticklabels(labels, fontweight='bold')
axes[1].legend(loc='upper right')
axes[1].set_ylim(0, 0.6)

plt.tight_layout()
output_path = './long_tail_analysis.png'
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"Plot successfully saved to: {output_path}")