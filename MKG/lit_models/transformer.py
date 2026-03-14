# transformer.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from .base import BaseLitModel
from transformers.optimization import get_linear_schedule_with_warmup

class TransformerLitModel(BaseLitModel):
    def __init__(self, model, args, tokenizer=None, data_config={}):
        super().__init__(model, args)
        self.save_hyperparameters(args)
        
        # ======== 损失函数定义 ========
        # 用于链接预测的损失函数
        self.link_prediction_loss = nn.CrossEntropyLoss()
        # 用于对比学习的温度参数
        self.temperature = nn.Parameter(torch.tensor(0.07))

        self.tokenizer = tokenizer
        self.__dict__.update(data_config) # 引入 num_relations, tokenizer_vocab_size 等

        # 调整模型的 token embedding 大小以适应我们新增的实体和关系 token
        self.model.resize_token_embeddings(len(self.tokenizer))


    def forward(self, **inputs):
        """
        覆盖基类的 forward, 直接将所有参数传递给 self.model
        """
        return self.model(**inputs)

    # =================================================================================
    # 核心修改: training_step
    # =================================================================================
    def training_step(self, batch, batch_idx):
        """
        根据 self.args.pretrain 的值，执行不同的训练逻辑。
        """
        # 移除标签，剩下的数据全部传给模型
        labels = batch.pop("labels")

        # ==================== 1. 预训练分支: 图文对比学习 ====================
        if self.args.pretrain:
            # 模型需要返回独立的文本和图片特征
            # 我们假设 self.model 在预训练模式下返回一个字典: {'text_features': ..., 'image_features': ...}
            outputs = self(**batch)
            text_features = outputs['text_features']
            image_features = outputs['image_features']

            # 归一化特征
            text_features = F.normalize(text_features, dim=-1)
            image_features = F.normalize(image_features, dim=-1)

            # 计算余弦相似度矩阵 [batch_size, batch_size]
            # 每个文本特征都要和 batch 内所有的图片特征计算相似度
            similarity_matrix = torch.matmul(text_features, image_features.t()) / self.temperature.exp()

            # 对比学习的标签是对角线 (batch 内第 i 个文本匹配第 i 个图片)
            # 标签的维度是 [batch_size]
            contrastive_labels = torch.arange(similarity_matrix.shape[0], device=self.device)

            # 计算图文对比损失 (InfoNCE loss)
            # 它等价于对相似度矩阵计算交叉熵
            loss = F.cross_entropy(similarity_matrix, contrastive_labels)
            
            self.log("Pretrain/loss", loss, prog_bar=True)
            self.log("Pretrain/temperature", self.temperature.exp(), prog_bar=True)

        # ==================== 2. 链接预测分支: 多模态知识图谱补全 ====================
        else:
            # 1. 模型返回三维的 logits: [batch_size, seq_len, vocab_size]
            logits_3d = self(**batch)
            
            # 2. ✅ 核心修改：我们使用第一个 token ([CLS]) 的输出来进行预测
            cls_logits = logits_3d[:, 0, :] # -> [batch_size, vocab_size]

            # 3. 从整个词汇表的 logits 中，只截取实体部分的 logits
            entity_logits = cls_logits[:, self.entity_id_st:self.entity_id_ed]
            
            # 4. 计算链接预测的交叉熵损失
            # labels 是尾实体在所有实体列表中的索引
            loss = self.link_prediction_loss(entity_logits, labels)
            
            self.log("Train/loss", loss)

        return loss


    def _eval(self, batch):
        labels = batch.pop("labels")
        
        # 1. ✅ 核心修改：同样，我们使用 [CLS] token 的输出来进行评估
        logits_3d = self(**batch)
        cls_logits = logits_3d[:, 0, :] # -> [batch_size, vocab_size]

        # 2. 截取实体部分的 logits
        logits = cls_logits[:, self.entity_id_st:self.entity_id_ed]
        bsz = logits.shape[0]

        # ... (诊断代码和后续的 filter_mask, sort 等逻辑保持不变)
        # if bsz > 0:
        #     print(f"\n[DIAGNOSTIC INFO] In _eval function:")
        #     print(f"  - Batch size (bsz): {bsz}")
        #     print(f"  - Labels tensor min value: {torch.min(labels)}")
        #     print(f"  - Labels tensor max value: {torch.max(labels)}")
        #     print(f"  - Logits shape (after CLS selection and entity slicing): {logits.shape}\n")

        filter_mask = torch.zeros_like(logits, dtype=torch.bool)
        filter_mask[torch.arange(bsz), labels] = True
        
        logits.masked_fill_(filter_mask, -1e9)
        _, sorted_indices = torch.sort(logits, dim=1, descending=True)
        _, ranks_tensor = torch.sort(sorted_indices, dim=1)
        
        ranks = ranks_tensor[torch.arange(bsz), labels].detach().cpu().numpy() + 1
        return {"ranks": ranks}

    # ... (validation_step, validation_epoch_end 等方法保持不变)

    def validation_step(self, batch, batch_idx):
        # 预训练时，我们只关心 loss，可以跳过复杂的评估
        if self.args.pretrain:
            return None
        return self._eval(batch)

    def validation_epoch_end(self, outputs):
        # 预训练时，outputs 会是 [None, None, ...], 直接返回
        if not outputs:
            return

        ranks = np.concatenate([out['ranks'] for out in outputs])
        
        hits1 = (ranks <= 1).mean()
        hits3 = (ranks <= 3).mean()
        hits10 = (ranks <= 10).mean()
        mrr = (1. / ranks).mean()

        self.log("Eval/mrr", mrr, prog_bar=True)
        self.log("Eval/hits1", hits1, prog_bar=True)
        self.log("Eval/hits3", hits3)
        self.log("Eval/hits10", hits10)
        self.log("Eval/mean_rank", ranks.mean())

    def test_step(self, batch, batch_idx):
        if self.args.pretrain:
            return None
        return self._eval(batch)

    def test_epoch_end(self, outputs):
        if not outputs:
            return
        # 与 validation_epoch_end 逻辑相同
        self.validation_epoch_end(outputs)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=self.args.weight_decay)
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=int(self.num_training_steps * self.args.warm_up_radio),
            num_training_steps=self.num_training_steps
        )
        return {
            "optimizer": optimizer, 
            "lr_scheduler": {
                'scheduler': scheduler,
                'interval': 'step',
                'frequency': 1,
            }
        }

    @staticmethod
    def add_to_argparse(parser):
        parser = BaseLitModel.add_to_argparse(parser)
        parser.add_argument("--label_smoothing", type=float, default=0.0)
        parser.add_argument("--bce", type=int, default=0)
        return parser