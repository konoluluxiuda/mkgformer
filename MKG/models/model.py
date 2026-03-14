# models/model.py
import torch 
from torch import nn
from .modeling_unimo import UnimoModel, UnimoOnlyMLMHead

class UnimoKGC(nn.Module):
    """
    一个包装 UnimoModel 的模型，用于处理我们的预训练和链接预测任务。
    """
    def __init__(self, vision_config, text_config, pretrain=False):
        super().__init__()
        self.pretrain = pretrain

        # 1. 核心编码器
        #    add_pooling_layer=True 使得 encoder 会有一个 pooler，我们可以利用它来获取 [CLS] token 的表示
        self.encoder = UnimoModel(vision_config, text_config, add_pooling_layer=True)

        # 2. 任务特定的头 (Heads)
        if self.pretrain:
            # === 用于预训练的投影头 ===
            # 将文本和图像的输出投影到相同的维度，用于计算对比损失
            projection_dim = 512  # 可以是任意维度，通常比 hidden_size 小
            self.text_projection = nn.Linear(text_config.hidden_size, projection_dim)
            self.image_projection = nn.Linear(vision_config.hidden_size, projection_dim)
        else:
            # === 用于链接预测的分类头 ===
            # UnimoOnlyMLMHead 内部包含了一个分类器，可以输出整个词汇表的分数
            self.cls = UnimoOnlyMLMHead(text_config)

    def forward(
        self,
        input_ids,
        attention_mask,
        pixel_values,
        token_type_ids=None, # 添加默认值为 None 的参数以匹配 batch
        labels=None,         # 接收 labels 但不使用，以匹配 batch
        **kwargs,            # 接收其他所有参数但不使用
    ):
        
        # ==================== 预训练分支 ====================
        if self.pretrain:
            # 在预训练时，我们不需要复杂的 aux 和 rcnn 输入
            # 我们假设 UnimoModel 可以处理这些值为 None 的情况
            outputs = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                pixel_values=pixel_values,
                aux_values=None,
                rcnn_values=None,
                return_dict=True,
            )
            
            # 从模型输出中获取文本和图像的池化特征 ([CLS] token 的特征)
            # pooler_output 是文本的 [CLS] token 经过一个 dense+tanh 层后的输出
            text_features = outputs.pooler_output
            
            # 对于图像，我们手动提取其特征
            # vision_embedding_output 的第一个 token 通常是 [CLS] token 的 embedding
            vision_embedding_output = self.encoder.vision_embeddings(pixel_values)
            # 这里我们简化一下，直接取 vision embedding 的平均值作为图像表示
            # 一个更标准的做法是使用 vision transformer 输出的 [CLS] token
            image_features = vision_embedding_output.mean(dim=1)

            # 将特征投影到共享空间
            text_features = self.text_projection(text_features)
            image_features = self.image_projection(image_features)
            
            return {
                'text_features': text_features,
                'image_features': image_features
            }

        # ==================== 链接预测分支 ====================
        else:
            outputs = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                pixel_values=pixel_values,
                aux_values=None, 
                rcnn_values=None,
                return_dict=True,
            )
            
            # sequence_output 的维度是 [batch_size, seq_len, hidden_size]
            sequence_output = outputs.last_hidden_state
            
            # 将序列输出传给分类头，得到整个词汇表的 logits
            # logits 的维度是 [batch_size, seq_len, vocab_size]
            logits = self.cls(sequence_output)
            
            return logits

    def resize_token_embeddings(self, new_num_tokens):
        """
        提供一个接口来调整底层 encoder 和分类头的 embedding 大小。
        """
        # 1. 调整底层 encoder 的 word embedding
        self.encoder.resize_token_embeddings(new_num_tokens)
        
        # 2. 如果是链接预测模式，需要彻底重建分类头(cls)的 decoder
        if not self.pretrain and hasattr(self.cls, 'predictions'):
            # 获取旧的 decoder 和 bias
            old_decoder = self.cls.predictions.decoder
            old_bias = self.cls.predictions.bias
            
            # 创建一个全新的、尺寸正确的 decoder
            new_decoder = nn.Linear(old_decoder.in_features, new_num_tokens, bias=False)
            new_decoder.to(old_decoder.weight.device, dtype=old_decoder.weight.dtype)
            
            # 创建一个全新的、尺寸正确的 bias
            new_bias = nn.Parameter(torch.zeros(new_num_tokens))
            new_bias.to(old_bias.device, dtype=old_bias.dtype)
            
            # 将旧的权重和偏置复制到新的层中
            num_tokens_to_copy = min(old_decoder.out_features, new_num_tokens)
            new_decoder.weight.data[:num_tokens_to_copy, :] = old_decoder.weight.data[:num_tokens_to_copy, :]
            new_bias.data[:num_tokens_to_copy] = old_bias.data[:num_tokens_to_copy]
            
            # ✅ 核心修改：用新的层替换旧的层
            self.cls.predictions.decoder = new_decoder
            self.cls.predictions.bias = new_bias
            
            # 确保 decoder 和 bias 保持关联
            self.cls.predictions.decoder.bias = self.cls.predictions.bias