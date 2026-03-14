# extract_features.py

import torch
import os
import json
import glob
from tqdm import tqdm
from PIL import Image
from torchvision import transforms
from transformers import AutoTokenizer, BertConfig
import numpy as np

# 导入我们已经写好的模型和数据处理类
from models.model import UnimoKGC
from models.modeling_clip import CLIPModel

# =================================================================
# 1. 配置和初始化
# =================================================================
@torch.no_grad() # 关键：我们只做前向传播，不需要计算梯度
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- 模型路径配置 ---
    data_dir = "dataset/HERB"
    bert_model_path = "models/bert-base-chinese"
    clip_model_path = "models/clip-vit-base-patch32"

    # --- 加载 Tokenizer ---
    tokenizer = AutoTokenizer.from_pretrained(bert_model_path)

    # --- 加载模型配置 ---
    text_config = BertConfig.from_pretrained(bert_model_path)
    clip_model = CLIPModel.from_pretrained(clip_model_path)
    vision_config = clip_model.config.vision_config

    # --- 实例化我们的 UnimoKGC 模型 ---
    # 注意：这里 pretrain=False，因为我们要用编码器的输出，而不是投影头的输出
    model = UnimoKGC(vision_config, text_config, pretrain=False)
    
    # 调整 embedding 大小以匹配 tokenizer (这一步很重要)
    entity_list = []
    with open(os.path.join(data_dir, "entities.txt"), 'r', encoding='utf-8') as f:
        entity_list = [line.strip() for line in f if line.strip()]
    
    relations_list = []
    with open(os.path.join(data_dir, "relations.txt"), 'r', encoding='utf-8') as f:
        relations_list = [line.strip() for line in f if line.strip()]
        
    tokenizer.add_special_tokens({"additional_special_tokens": entity_list + relations_list})
    model.resize_token_embeddings(len(tokenizer))
    
    model.to(device)
    model.eval() # 切换到评估模式

    # --- 准备数据 ---
    # 加载所有实体的文本描述
    entity_text_map = {}
    with open(os.path.join(data_dir, "entity2textlong_updated.txt"), 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split('\t', 1)
            if len(parts) == 2:
                entity_text_map[parts[0]] = parts[1]

    # 加载实体模态信息
    with open(os.path.join(data_dir, "entity2modality.json"), 'r', encoding='utf-8') as f:
        entity2modality = json.load(f)

    # 准备图片变换器
    image_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711])
    ])
    image_root = "dataset/HERB-images"

    # =================================================================
    # 2. 遍历所有实体，提取特征
    # =================================================================
    all_entity_embeddings = {}

    print("Extracting features for all entities...")
    for entity_name in tqdm(entity_list):
        # --- 准备文本输入 ---
        text_description = entity_text_map.get(entity_name, entity_name) # 如果没描述，就用名字
        text_inputs = tokenizer(text_description, return_tensors="pt", max_length=128, padding="max_length", truncation=True).to(device)

        # --- 准备图像输入 ---
        # pixel_values = torch.zeros(1, 3, 224, 224).to(device) # 默认零占位符
        # if "image" in entity2modality.get(entity_name, []):
        #     img_folder = os.path.join(image_root, entity_name)
        #     if os.path.isdir(img_folder):
        #         image_files = glob.glob(os.path.join(img_folder, "*.jpg"))
        #         if image_files:
        #             try:
        #                 image = Image.open(image_files[0]).convert("RGB")
        #                 pixel_values = image_transform(image).unsqueeze(0).to(device)
        #             except Exception:
        #                 pass # 图片加载失败，则使用零占位符
        pixel_values = torch.zeros(1, 3, 224, 224).to(device)
        
        # --- 模型前向传播 ---
        # 调用模型的底层编码器
        encoder_outputs = model.encoder(
            input_ids=text_inputs.input_ids,
            attention_mask=text_inputs.attention_mask,
            pixel_values=pixel_values,
            return_dict=True
        )
        
        # 提取 [CLS] token 的输出作为该实体的特征向量
        # encoder_outputs.last_hidden_state 的形状是 [1, seq_len, hidden_size]
        cls_embedding = encoder_outputs.last_hidden_state[:, 0, :].squeeze().cpu().numpy()
        all_entity_embeddings[entity_name] = cls_embedding

    # =================================================================
    # 3. 保存结果
    # =================================================================
    # 将实体名称列表和 embedding 矩阵分开保存
    output_path = "output/HERB"
    os.makedirs(output_path, exist_ok=True)
    
    # 确保 embedding 矩阵的顺序与实体列表的顺序完全一致
    embedding_matrix = np.array([all_entity_embeddings[name] for name in entity_list])

    # np.save(os.path.join(output_path, "entity_embeddings.npy"), embedding_matrix)
    np.save(os.path.join(output_path, "entity_embeddings_text_only.npy"), embedding_matrix) 
    with open(os.path.join(output_path, "entity_list.json"), 'w', encoding='utf-8') as f:
        json.dump(entity_list, f, ensure_ascii=False, indent=4)

    print(f"Feature extraction complete!")
    print(f"Embedding matrix shape: {embedding_matrix.shape}")
    print(f"Embeddings saved to: {os.path.join(output_path, 'entity_embeddings.npy')}")
    print(f"Entity list saved to: {os.path.join(output_path, 'entity_list.json')}")

if __name__ == "__main__":
    main()