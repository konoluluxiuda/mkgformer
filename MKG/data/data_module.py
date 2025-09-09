import os
import torch
import random
from PIL import Image
from enum import Enum
from os import listdir
from dataclasses import dataclass
from typing import Any, Optional, Union
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from transformers.models.clip import CLIPProcessor
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from .base_data_module import BaseDataModule
from .processor import KGProcessor, get_dataset
from .processor import KGProcessor, MultiprocessingEncoder, KGCDataset
# 屏蔽 transformers 警告
from transformers import logging
logging.set_verbosity_error()


# CLIP 处理器
aux_size, rcnn_size = 128, 64
clip_processor = CLIPProcessor.from_pretrained('/opt/workspace/MKGformer/MKG/models/clip-vit-base-patch32')
aux_processor = CLIPProcessor.from_pretrained('/opt/workspace/MKGformer/MKG/models/clip-vit-base-patch32')
aux_processor.feature_extractor.size, aux_processor.feature_extractor.crop_size = aux_size, aux_size
rcnn_processor = CLIPProcessor.from_pretrained('/opt/workspace/MKGformer/MKG/models/clip-vit-base-patch32')
rcnn_processor.feature_extractor.size, rcnn_processor.feature_extractor.crop_size = rcnn_size, rcnn_size


class ExplicitEnum(Enum):
    @classmethod
    def _missing_(cls, value):
        raise ValueError(
            f"{value} is not a valid {cls.__name__}, please select one of {list(cls._value2member_map_.keys())}"
        )


class PaddingStrategy(ExplicitEnum):
    LONGEST = "longest"
    MAX_LENGTH = "max_length"
    DO_NOT_PAD = "do_not_pad"


@dataclass
class DataCollatorForSeq2Seq:
    tokenizer: PreTrainedTokenizerBase
    model: Optional[Any] = None
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    label_pad_token_id: int = -100
    return_tensors: str = "pt"
    num_labels: int = 0
    task_name: str = None
    entity_img_path: str = None
    entity_img_files: Optional[Any] = None

def __call__(self, features, return_tensors=None):
    if return_tensors is None:
        return_tensors = self.return_tensors

    # 提取 labels 和 entities（不修改 features 本身）
    labels = [f["labels"] for f in features] if "labels" in features[0] else None
    label = [f["label"] for f in features]
    entities = [f["entity"] for f in features] if "entity" in features[0] else None

    # 先复制 keys，避免 pop
    features_keys = {}
    for k in list(features[0].keys()):
        if k in ["input_ids", "attention_mask", "token_type_ids", "labels", "label", "entity"]:
            continue
        features_keys[k] = [f[k] for f in features]

    # 处理 labels -> one-hot
    if labels is not None:
        bsz = len(labels)
        with torch.no_grad():
            new_labels = torch.zeros(bsz, self.num_labels)
            for i, l in enumerate(labels):
                # 如果 l 是 0维 tensor，先转成 int
                if isinstance(l, torch.Tensor) and l.dim() == 0:
                    l = int(l.item())

                if isinstance(l, int):
                    new_labels[i][l] = 1
                elif isinstance(l, (list, tuple)):
                    for j in l:
                        # 如果 j 也是 tensor，则转成 int
                        if isinstance(j, torch.Tensor) and j.dim() == 0:
                            j = int(j.item())
                        new_labels[i][j] = 1
                else:
                    raise ValueError(f"Unexpected label type: {type(l)} -> {l}")
            labels = new_labels

    # --- pad 前检查 ---
    tokenizer_fields = ["input_ids", "attention_mask", "token_type_ids"]
    for k, v in features[0].items():
        if k not in tokenizer_fields and k not in ["labels", "label", "entity"]:
            if isinstance(v, str):
                raise ValueError(
                    f"❌ 非 tokenizer 字段 '{k}' 含字符串: type={type(v)} value={v[:50]}"
                )

    # pad 文本
    features = self.tokenizer.pad(
        features,
        padding=self.padding,
        max_length=self.max_length,
        pad_to_multiple_of=self.pad_to_multiple_of,
        return_tensors=return_tensors
    )

    if labels is not None:
        features["labels"] = labels
    features["label"] = torch.tensor(label)
    features.update(features_keys)

    # 加载实体图像
    pixel_images = []
    if entities is not None and "modality" in features_keys:
        for entity, modality in zip(entities, features_keys["modality"]):
            if modality == "image" and entity:
                img_path = entity
                if os.path.exists(img_path):
                    img = Image.open(img_path).convert("RGB")
                    img_tensor = clip_processor(images=img, return_tensors="pt")["pixel_values"].squeeze()
                else:
                    img_tensor = torch.zeros((3, 224, 224))
                pixel_images.append(img_tensor)
            else:
                pixel_images.append(torch.zeros((3, 224, 224)))
        features["pixel_values"] = torch.stack(pixel_images)

    return features




class KGC(BaseDataModule):
    def __init__(self, args, model):
        super().__init__(args)
        self.tokenizer = AutoTokenizer.from_pretrained(self.args.model_name_or_path, use_fast=False)
        self.processor = KGProcessor(self.tokenizer, args)
        self.label_list = self.processor.get_labels(args.data_dir)

        entity_list = self.processor.get_entities(args.data_dir)
        self.tokenizer.add_special_tokens({"additional_special_tokens": entity_list})

        # HERB 图像路径
        entity_img_path = os.path.join(args.data_dir, "../HERB-images")
        entity_img_files = listdir(entity_img_path)
        self.sampler = DataCollatorForSeq2Seq(
            self.tokenizer,
            model=model,
            label_pad_token_id=self.tokenizer.pad_token_id,
            max_length=self.args.max_seq_length,
            num_labels=len(entity_list),
            task_name="herb",
            entity_img_path=entity_img_path,
            entity_img_files=entity_img_files
        )

        relations_tokens = self.processor.get_relations(args.data_dir)
        self.num_relations = len(relations_tokens)
        self.tokenizer.add_special_tokens({'additional_special_tokens': relations_tokens})

        self.encoder = MultiprocessingEncoder(self.tokenizer, max_seq_length=self.args.max_seq_length)

    def setup(self, stage=None):
        # 1. 读取原始 triples
        train_triples = get_dataset(self.args, split="train")
        val_triples = get_dataset(self.args, split="dev")
        test_triples = get_dataset(self.args, split="test")

        # 2. 转换成 InputExample
        train_examples = self.processor._create_examples(train_triples, "train")
        val_examples = self.processor._create_examples(val_triples, "dev")
        test_examples = self.processor._create_examples(test_triples, "test")

        # 3. 转换成 InputFeatures
        train_features = self.encoder.encode_lines(train_examples)
        val_features = self.encoder.encode_lines(val_examples)
        test_features = self.encoder.encode_lines(test_examples)

        # 4. 包装成 Dataset
        self.data_train = KGCDataset(train_features)
        self.data_val = KGCDataset(val_features)
        self.data_test = KGCDataset(test_features)

    def get_config(self):
        """
        返回数据相关的配置，供 LitModel 使用（例如实体/关系在 tokenizer 中的 id 范围等）。
        保证返回的键与 lit_model 期望的一致。
        """
        cfg = {}
        # 关系/实体 id 范围（在 tokenizer 中追加 special tokens 后的 index）
        # 这些属性应在 __init__ 中被设置：relation_id_st, relation_id_ed, entity_id_st, entity_id_ed, num_relations
        if hasattr(self, "relation_id_st"):
            cfg["relation_id_st"] = self.relation_id_st
        if hasattr(self, "relation_id_ed"):
            cfg["relation_id_ed"] = self.relation_id_ed
        if hasattr(self, "entity_id_st"):
            cfg["entity_id_st"] = self.entity_id_st
        if hasattr(self, "entity_id_ed"):
            cfg["entity_id_ed"] = self.entity_id_ed
        if hasattr(self, "num_relations"):
            cfg["num_relations"] = self.num_relations

        # 还可以返回 tokenizer / image path 等（视 lit_model 需要）
        cfg["tokenizer_vocab_size"] = len(self.tokenizer)
        cfg["image_root"] = getattr(self, "entity_img_path", None)

        return cfg
    
    
    def train_dataloader(self):
        return DataLoader(
            self.data_train,
            batch_size=self.args.batch_size,
            shuffle=self.args.pretrain,
            num_workers=self.num_workers,
            pin_memory=False,
            collate_fn=self.sampler
        )

    def val_dataloader(self):
        return DataLoader(
            self.data_val,
            batch_size=self.args.eval_batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=False,
            collate_fn=self.sampler
        )

    def test_dataloader(self):
        return DataLoader(
            self.data_test,
            batch_size=self.args.eval_batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=False,
            collate_fn=self.sampler
        )