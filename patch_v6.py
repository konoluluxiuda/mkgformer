import re

with open('fig/draw_anti_hub_real_data.py', 'r', encoding='utf-8') as f:
    content = f.read()

# 1. MIN_SIM
content = content.replace(
    "MIN_SIM = 0.8# 【修正】调高 \tau 阈值，只有高度相似才能满额，赋予真实方差",
    "MIN_SIM = 0.0   # 【修正】移除硬阈值，防止中心hub孤立"
)

# 2. b图孤立节点
old_b = """# ---------- (b) 剪枝后子图 (调小箭头并曲线化) ----------
ax2 = axes[0, 1]
# 【修正】即使修剪后 hub 的边数量发生了剧变，也强制把红色中心点和原本的邻居框架画出来，保持视觉对齐
nx.draw_networkx_nodes(sub_raw, pos, ax=ax2, nodelist=[n for n in sampled_nodes if n != hub_name], node_color=COLOR_PRIM, node_size=80, alpha=0.9, edgecolors='white')
nx.draw_networkx_nodes(sub_raw, pos, ax=ax2, nodelist=[hub_name], node_color=COLOR_TARGET, node_size=450, edgecolors='black', linewidths=1.5)"""

new_b = """# ---------- (b) 剪枝后子图 (调小箭头并曲线化) ----------
ax2 = axes[0, 1]
# 【修正】剔除孤立节点，只保留有连线的节点，使图b更加干净清爽
active_nodes_b = [n for n in sampled_nodes if sub_pruned.degree(n) > 0 or n == hub_name]

nx.draw_networkx_nodes(sub_raw, pos, ax=ax2, nodelist=[n for n in active_nodes_b if n != hub_name], node_color=COLOR_PRIM, node_size=80, alpha=0.9, edgecolors='white')
nx.draw_networkx_nodes(sub_raw, pos, ax=ax2, nodelist=[hub_name] if hub_name in active_nodes_b else [], node_color=COLOR_TARGET, node_size=450, edgecolors='black', linewidths=1.5)"""

content = content.replace(old_b, new_b)

# 3. 删除虚线和文字
old_d = """ax4.grid(True, axis='y', ls="--", alpha=0.3)

# 【修改：画优雅的垂直红虚线代替狗皮膏药】
ax4.axvline(x=MAX_NEIGHBORS, color=COLOR_TARGET, linestyle='--', linewidth=2, alpha=0.8)
ax4.text(MAX_NEIGHBORS - 0.4, max(y_out) * 0.90, f"Capacity K={MAX_NEIGHBORS}", 
         color=COLOR_TARGET, fontsize=14, ha='right')

plt.tight_layout(pad=3.0)
save_path_pdf = os.path.join(CURRENT_DIR, "anti_hub_Ginseng_Optimized_V4.pdf")
save_path_png = os.path.join(CURRENT_DIR, "anti_hub_Ginseng_Optimized_V4.png")"""

new_d = """ax4.grid(True, axis='y', ls="--", alpha=0.3)

plt.tight_layout(pad=3.0)
save_path_pdf = os.path.join(CURRENT_DIR, "anti_hub_Ginseng_Optimized_V6.pdf")
save_path_png = os.path.join(CURRENT_DIR, "anti_hub_Ginseng_Optimized_V6.png")"""

content = content.replace(old_d, new_d)

with open('fig/draw_anti_hub_real_data.py', 'w', encoding='utf-8') as f:
    f.write(content)
