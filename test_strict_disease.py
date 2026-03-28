import pandas as pd
import os

newherb_herbs = set(pd.read_csv('MKG/dataset/NEWHERB/entities/herb.csv')['id'])
kdhr_h2d = pd.read_csv('KDHR/KG/relation/herbTOdisease.csv')

# Note: KDHR herbTOdisease format: :START_ID -> herb, :END_ID -> disease
# It might have column names :START_ID, :END_ID, relation, :TYPE
if ':START_ID' in kdhr_h2d.columns:
    kdhr_h2d = kdhr_h2d[[':START_ID', ':END_ID']].rename(columns={':START_ID': 'herb', ':END_ID': 'disease'})
else:
    kdhr_h2d = kdhr_h2d.rename(columns={kdhr_h2d.columns[0]: 'herb', kdhr_h2d.columns[1]: 'disease'})

# group by disease
disease_to_herbs = kdhr_h2d.groupby('disease')['herb'].apply(set).to_dict()

strict_diseases = []
for d, h_set in disease_to_herbs.items():
    if h_set.issubset(newherb_herbs):
        strict_diseases.append(d)
        
print(f"Total diseases in KDHR graph: {len(disease_to_herbs)}")
print(f"Total diseases with ALL their KDHR herbs in NEWHERB: {len(strict_diseases)}")
