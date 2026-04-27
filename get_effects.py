import pandas as pd
import os

base_dir = '/home/zry/workspace/mkgformer/MKG/dataset/NEWHERB/'

# Load entities
try:
    herbs = pd.read_csv(os.path.join(base_dir, 'entities/herb.csv'))
    effects = pd.read_csv(os.path.join(base_dir, 'entities/effect.csv'))
    # Try properties or other things if necessary, but effect.csv is requested
except Exception as e:
    print("Error loading:", e)

# Load relationships
try:
    herb_to_effect = pd.read_csv(os.path.join(base_dir, 'relation/herbTOeffect.csv'))
except Exception as e:
    print("Error loading rel:", e)

# Standardize columns
herb_to_effect.columns = ['herb_id', 'effect_id']

target_herbs = ['鸦胆子', '牛膝', '红参', '灵芝', '泽兰', '地黄', '天冬', '山茱萸', '山药', '大枣', '白术', '马齿苋', '苍耳子', '商陆', '党参']

# Map
for herb_name in target_herbs:
    herb_row = herbs[herbs['name'] == herb_name]
    if herb_row.empty:
        print(f"{herb_name}: Not found in herbs list.")
        continue
    h_id = herb_row.iloc[0]['id']
    
    e_ids = herb_to_effect[herb_to_effect['herb_id'] == h_id]['effect_id']
    
    e_names = effects[effects['id'].isin(e_ids)]['name'].tolist()
    print(f"【{herb_name}】: {', '.join(e_names)}")

