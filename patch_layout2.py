import re

with open('fig/draw_anti_hub_real_data.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Remove the red text in plot (b)
content = re.sub(
    r"ax2\.text\(pos.*?Hub Inbound/Outbound Restricted \(\$\\tau \\geq 0\.8\)\'.*?edgecolor='none'\)\)",
    "",
    content,
    flags=re.DOTALL
)

# Adjust layout: Use Kamada-Kawai or a better spring layout seeded from the sub_pruned graph
old_layout_lines = "pos = nx.spring_layout(sub_raw, seed=42, k=0.6)"
new_layout_lines = """# 根据剪枝图计算引力模型，使得稠密的点能够由于剪枝后的拓扑张开，稀疏的点能合理分布
pos = nx.spring_layout(sub_pruned, seed=15, k=1.0, iterations=150)
pos_extra = nx.spring_layout(sub_raw, seed=15, k=1.0, iterations=150)
for node in sub_raw.nodes():
    if node not in pos:
        pos[node] = pos_extra[node]"""
content = content.replace(old_layout_lines, new_layout_lines)

with open('fig/draw_anti_hub_real_data.py', 'w', encoding='utf-8') as f:
    f.write(content)
