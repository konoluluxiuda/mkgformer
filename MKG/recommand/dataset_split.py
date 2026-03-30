import torch
import os
from config import Config

class SplitGraphDataManager:
    def __init__(self):
        # 动态使用 Config 里指定的文件夹（保持与 train.py 完全一致的数据源）
        self.data_dir = Config.REC_DATA_DIR
        
    def load_split_data(self):
        print(f"Loading data from {self.data_dir}...")
        
        # 1. 加载基础数据
        rec_data = torch.load(os.path.join(self.data_dir, 'rec_data.pt'))
        edge_index = torch.load(os.path.join(self.data_dir, 'edge_index.pt'))
        edge_type = torch.load(os.path.join(self.data_dir, 'edge_type.pt'))
        
        num_nodes = rec_data['num_nodes']
        train_dict = rec_data['train_dict']
        test_dict = rec_data['test_dict']
        herb_indices = rec_data['herb_indices']
        
        # 2. 物理拆分图结构 (基于 edge_type)
        # 根据 preprocess_paper_graph.py 的定义:
        # 0-9: S-H Graph (原始治疗关系 + 属性关系) -> 实际上属性关系不属于SH二部图，但为了简便通常保留
        # 10: H-H Graph (草药协作)
        # 11: S-S Graph (疾病协作)
        
        print("Splitting graphs...")
        
        # --- S-H Graph (二部图) ---
        # 保留治疗关系 (type 0 和 5) 以及其他属性关系作为基础交互
        # KDHR 中 S-H 主要指治疗关系
        sh_mask = (edge_type < 10) 
        edge_index_sh = edge_index[:, sh_mask]
        print(f"  > S-H Graph edges: {edge_index_sh.shape[1]}")
        
        # --- H-H Graph (草药协作) ---
        hh_mask = (edge_type == 10)
        edge_index_hh = edge_index[:, hh_mask]
        print(f"  > H-H Graph edges: {edge_index_hh.shape[1]}")
        
        # --- S-S Graph (疾病协作) ---
        ss_mask = (edge_type == 11)
        edge_index_ss = edge_index[:, ss_mask]
        print(f"  > S-S Graph edges: {edge_index_ss.shape[1]}")
        
        return {
            'num_nodes': num_nodes,
            'herb_indices': herb_indices,
            'train_dict': train_dict,
            'test_dict': test_dict,
            'sh_graph': edge_index_sh,
            'hh_graph': edge_index_hh,
            'ss_graph': edge_index_ss
        }

    def load_attributes(self):
        # 复用之前的属性加载逻辑
        attr_path = os.path.join(self.data_dir, 'node_attributes.pt')
        if os.path.exists(attr_path):
            return torch.load(attr_path)
        return None