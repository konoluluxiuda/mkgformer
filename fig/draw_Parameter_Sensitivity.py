import matplotlib.pyplot as plt
import numpy as np

# 设置学术画图风格
plt.style.use('seaborn-v0_8-whitegrid')
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

# ==========================================
# (a) 真实的 K 参数敏感性分析数据
# ==========================================
K_values = [5, 10, 15, 20]
# 真实数据
K_F1 = [0.1723, 0.1727, 0.1649, 0.1646]
K_Recall = [0.3294, 0.3294, 0.3110, 0.3101]

ax1.plot(K_values, K_F1, marker='o', color='#1f77b4', linewidth=2.5, markersize=8, label='F1@10')
ax1.set_xlabel('Truncation Size $K_{Herb}$', fontsize=13)
ax1.set_ylabel('F1@10', color='#1f77b4', fontsize=13)
ax1.tick_params(axis='y', labelcolor='#1f77b4')
ax1.set_xticks(K_values)

ax1_twin = ax1.twinx()
ax1_twin.plot(K_values, K_Recall, marker='s', color='#ff7f0e', linewidth=2.5, markersize=8, linestyle='--', label='Recall@10')
ax1_twin.set_ylabel('Recall@10', color='#ff7f0e', fontsize=13)
ax1_twin.tick_params(axis='y', labelcolor='#ff7f0e')

ax1.set_title('(a) Sensitivity to Truncation Size $K$', fontsize=14, pad=10)
ax1.grid(True, linestyle=':', alpha=0.6)

# ==========================================
# (b) 真实的 λ 参数敏感性分析数据
# ==========================================
lambda_values = ['0.01', '0.05', '0.1', '0.2', '0.5']
# 真实数据
L_F1 = [0.1668, 0.1712, 0.1732, 0.1714, 0.1683]
L_Recall = [0.3166, 0.3280, 0.3297, 0.3281, 0.3214]

ax2.plot(lambda_values, L_F1, marker='o', color='#1f77b4', linewidth=2.5, markersize=8, label='F1@10')
ax2.set_xlabel('Contrastive Base Weight $\lambda$', fontsize=13)
ax2.set_ylabel('F1@10', color='#1f77b4', fontsize=13)
ax2.tick_params(axis='y', labelcolor='#1f77b4')

ax2_twin = ax2.twinx()
ax2_twin.plot(lambda_values, L_Recall, marker='s', color='#ff7f0e', linewidth=2.5, markersize=8, linestyle='--', label='Recall@10')
ax2_twin.set_ylabel('Recall@10', color='#ff7f0e', fontsize=13)
ax2_twin.tick_params(axis='y', labelcolor='#ff7f0e')

ax2.set_title('(b) Sensitivity to Contrastive Weight $\lambda$', fontsize=14, pad=10)
ax2.grid(True, linestyle=':', alpha=0.6)

# 合并双 Y 轴图例
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax1_twin.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc='lower center', fontsize=11, framealpha=0.9)

lines3, labels3 = ax2.get_legend_handles_labels()
lines4, labels4 = ax2_twin.get_legend_handles_labels()
ax2.legend(lines3 + lines4, labels3 + labels4, loc='lower center', fontsize=11, framealpha=0.9)

plt.tight_layout()
plt.savefig('Parameter_Sensitivity_Final.png', dpi=300, bbox_inches='tight')
print("✅ Saved Parameter_Sensitivity_Final.png! Graph is ready for your paper.")