import csv
import os

base_dir = '/home/zry/workspace/mkgformer/MKG/dataset/NEWHERB/'

herbs = {}
with open(os.path.join(base_dir, 'entities/herb.csv'), encoding='utf-8') as f:
    for row in csv.reader(f):
        if row[0] == 'id': continue
        herbs[row[1]] = row[0]

effects = {}
with open(os.path.join(base_dir, 'entities/effect.csv'), encoding='utf-8') as f:
    for row in csv.reader(f):
        if row[0] == 'id': continue
        effects[row[0]] = row[1]

herb_to_effect = {}
with open(os.path.join(base_dir, 'relation/herbTOeffect.csv'), encoding='utf-8') as f:
    for row in csv.reader(f):
        if row[0] == ':START_ID': continue
        h_id, e_id = row[0], row[1]
        if h_id not in herb_to_effect: herb_to_effect[h_id] = []
        herb_to_effect[h_id].append(e_id)

target_herbs = ['鸦胆子', '牛膝', '红参', '灵芝', '泽兰', '地黄', '天冬', '山茱萸', '山药', '大枣', '白术', '马齿苋', '苍耳子', '商陆', '党参']

for herb_name in target_herbs:
    if herb_name not in herbs:
        print(f"【{herb_name}】: Not found")
        continue
    h_id = herbs[herb_name]
    e_ids = herb_to_effect.get(h_id, [])
    e_names = [effects[e] for e in e_ids if e in effects]
    print(f"【{herb_name}】: {', '.join(e_names)}")

