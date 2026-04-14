import re

with open('fig/draw_anti_hub_real_data.py', 'r', encoding='utf-8') as f:
    content = f.read()

# 1. Remove the red text in (b)
content = re.sub(
    r"ax2\.text\(pos.*?Hub Inbound/Outbound Restricted \(\$\\tau \\geq 0\.8\)\'.*?edgecolor='none'\)\)",
    "",
    content,
    flags=re.DOTALL
)

# 2. Adjust layout for (b) to be more scientific and clean
# Let's adjust node positions or just tweak layout parameter
old_pos = "pos = nx.spring_layout(sub_raw, seed=42, k=0.6)"
new_pos = "pos = nx.kamada_kawai_layout(sub_raw)" # Kamada-Kawai often looks more balanced and neat
content = content.replace(old_pos, new_pos)

# Try Spring layout with better distances if KK is not found/not preferred
# Let's actually use spring_layout but with better k, like k=1.0 or iterations
# Just use spring_layout(sub_raw, seed=45, k=0.8, iterations=100)
new_pos2 = "pos = nx.spring_layout(sub_raw, seed=123, k=0.8, iterations=100)"
content = content.replace(new_pos, new_pos2) # in case kamada was replaced
content = content.replace(old_pos, new_pos2)

with open('fig/draw_anti_hub_real_data.py', 'w', encoding='utf-8') as f:
    f.write(content)
