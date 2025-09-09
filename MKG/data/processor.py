import os
import sys
import json
import torch
import pickle
import logging
from tqdm import tqdm
from functools import partial
from collections import defaultdict
from dataclasses import dataclass, asdict
from torch.utils.data import Dataset
from transformers import AutoTokenizer

logger = logging.getLogger(__name__)


@dataclass
class InputFeatures:
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    labels: torch.Tensor = None
    label: torch.Tensor = None
    en: torch.Tensor = 0
    rel: torch.Tensor = 0
    entity: torch.Tensor = None
    entity_text: str = None
    entity_image: str = None
    modality: str = None


def get_dataset(args=None, processor=None, label_list=None, tokenizer=None, split="train"):
    import os
    data_dir = args.data_dir if args else processor.args.data_dir
    triple_path = os.path.join(data_dir, f"{split}.tsv")
    if not os.path.exists(triple_path):
        raise FileNotFoundError(f"{triple_path} 不存在")

    triples = []
    with open(triple_path, "r", encoding="utf-8") as f:
        for line in f:
            h, r, t = line.strip().split("\t")
            triples.append((h, r, t))

    return triples


class InputExample:
    def __init__(self, guid, text_a, text_b=None, text_c=None,
                 label=None, real_label=None, en=None, rel=None,
                 entity=None, entity_text=None, entity_image=None, modality="text"):
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.text_c = text_c
        self.label = label
        self.real_label = real_label
        self.en = en
        self.rel = rel
        self.entity = entity
        self.entity_text = entity_text
        self.entity_image = entity_image
        self.modality = modality


class KGProcessor:
    """Processor for knowledge graph data set with text and image modalities."""
    def __init__(self, tokenizer, args):
        self.tokenizer = tokenizer
        self.args = args
        self.image_dir = os.path.join(args.data_dir, "../HERB-images")
        self.entity_path = os.path.join(args.data_dir, "entity2textlong_updated.txt")
        # 加载实体模态信息
        with open(os.path.join(args.data_dir, "entity2modality.json"), "r", encoding="utf-8") as f:
            self.ent2modality = json.load(f)

    def get_relations(self, data_dir):
        """Gets all relations and maps them to special tokens."""
        relation_file = os.path.join(data_dir, "relations.txt")
        relations = []
        if os.path.exists(relation_file):
            with open(relation_file, "r", encoding="utf-8") as f:
                for line in f:
                    rel = line.strip().split("\t")[0]
                    if rel:
                        relations.append(rel)
        # 将关系映射到特殊 token
        rel2token = {rel: f"[RELATION_{i}]" for i, rel in enumerate(relations)}
        return list(rel2token.values())

    def get_labels(self, data_dir):
        """Gets labels for the triples, if needed."""
        label_file = os.path.join(data_dir, "relation2text.txt")
        labels = []
        if os.path.exists(label_file):
            with open(label_file, "r", encoding="utf-8") as f:
                for line in f:
                    labels.append(line.strip().split("\t")[-1])
        return labels

    def get_entities(self, data_dir):
        """Reads all entities and maps them to special tokens."""
        if not os.path.exists(self.entity_path):
            raise FileNotFoundError(f"{self.entity_path} not found!")
        entities = []
        with open(self.entity_path, "r", encoding="utf-8") as f:
            for line in f:
                ent = line.strip().split("\t")[0]
                if ent:
                    entities.append(ent)
        ent2token = {ent: f"[ENTITY_{i}]" for i, ent in enumerate(entities)}
        return list(ent2token.values())

    def _create_examples(self, lines, set_type):
        """Generates InputExample objects for triples, supporting text+image modalities."""
        examples = []
        ent2text = {}
        ent2image = {}

        # 读取实体文本
        with open(self.entity_path, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split("\t")
                if len(parts) >= 2:
                    ent, desc = parts[0], parts[1]
                    ent2text[ent] = desc

        # 准备实体图像 & 更新 ent2modality 为字符串
        for ent in ent2text.keys():
            modality_list = self.ent2modality.get(ent, ["text"])
            if "image" in modality_list:
                modality = "image"
                herb_dir = os.path.join(self.image_dir, ent)
                if os.path.exists(herb_dir) and len(os.listdir(herb_dir)) > 0:
                    ent2image[ent] = os.path.join(herb_dir, os.listdir(herb_dir)[0])
                else:
                    ent2image[ent] = "[NO_IMAGE]"
            else:
                modality = "text"
                ent2image[ent] = None

            # 将 ent2modality 更新为字符串形式
            self.ent2modality[ent] = modality

        # 创建 InputExample
        for i, line in enumerate(lines):
            head, relation, tail = line[0], line[1], line[2]
            head_mod = self.ent2modality.get(head, "text")
            tail_mod = self.ent2modality.get(tail, "text")

            guid = f"{set_type}-{i}"

            # head 实体为主
            examples.append(
                InputExample(
                    guid=guid,
                    text_a=ent2text.get(head, "[NO_TEXT]"),
                    text_b=relation,
                    text_c=ent2text.get(tail, "[NO_TEXT]"),
                    entity=head,
                    entity_text=ent2text.get(head, "[NO_TEXT]"),
                    entity_image=ent2image.get(head, "[NO_IMAGE]"),
                    modality=head_mod
                )
            )
            # tail 实体为主
            examples.append(
                InputExample(
                    guid=guid + "_tail",
                    text_a=ent2text.get(tail, "[NO_TEXT]"),
                    text_b=relation,
                    text_c=ent2text.get(head, "[NO_TEXT]"),
                    entity=tail,
                    entity_text=ent2text.get(tail, "[NO_TEXT]"),
                    entity_image=ent2image.get(tail, "[NO_IMAGE]"),
                    modality=tail_mod
                )
            )

        return examples



class KGCDataset(Dataset):
    """Knowledge Graph Dataset that returns dicts, compatible with PyTorch DataLoader."""
    def __init__(self, features): 
        self.features = features

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        f = self.features[idx]
        return {
            "input_ids": f.input_ids,
            "attention_mask": f.attention_mask,
            "labels": f.labels if f.labels is not None else torch.tensor(-100),
            "label": f.label if f.label is not None else torch.tensor(-100),
            "en": f.en if f.en is not None else torch.tensor(0),
            "rel": f.rel if f.rel is not None else torch.tensor(0),
            "entity": f.entity if f.entity is not None else torch.tensor(0),
            "entity_text": f.entity_text if f.entity_text is not None else "[NO_TEXT]",
            "entity_image": f.entity_image if f.entity_image is not None else "[NO_IMAGE]",
            "modality": f.modality if f.modality is not None else "text"
        }


class MultiprocessingEncoder:
    """支持多线程将 InputExample 转换为 InputFeatures"""
    def __init__(self, tokenizer, max_seq_length=128):
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length

    def convert_examples_to_features(self, example: InputExample):
        # 文本序列拼接
        if example.modality == "text":
            input_text_a = example.entity_text
            input_text_b = example.text_b + " " + example.text_c
        else:
            # image 模态可以用图像路径作为文本描述 placeholder
            img_text = f"[IMAGE:{example.entity_image}]" if example.entity_image else "[NO_IMAGE]"
            input_text_a = img_text
            input_text_b = example.text_b + " " + example.text_c

        inputs = self.tokenizer(
            input_text_a,
            input_text_b,
            truncation="longest_first",
            max_length=self.max_seq_length,
            padding="max_length",
            add_special_tokens=True
        )

        features = InputFeatures(
            input_ids=torch.tensor(inputs["input_ids"]),
            attention_mask=torch.tensor(inputs["attention_mask"]),
            labels=torch.tensor(example.label) if example.label is not None else None,
            label=torch.tensor(example.real_label) if example.real_label is not None else None,
            en=example.en,
            rel=example.rel,
            entity=example.entity,
            entity_text=example.entity_text,
            entity_image=example.entity_image,
            modality=example.modality
        )
        return features

    def encode_lines(self, examples):
        features = []
        for ex in examples:
            features.append(self.convert_examples_to_features(ex))
        return features
