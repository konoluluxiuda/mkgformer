import os
import torch
import argparse
import importlib
import numpy as np
import pytorch_lightning as pl
from transformers import BertConfig, BertModel
from models.modeling_clip import CLIPModel
from models.model import UnimoKGC
from pytorch_lightning import callbacks

os.environ["TOKENIZERS_PARALLELISM"] = "false"


def _import_class(module_and_class_name: str) -> type:
    """动态导入类"""
    module_name, class_name = module_and_class_name.rsplit(".", 1)
    module = importlib.import_module(module_name)
    return getattr(module, class_name)


def _setup_parser():
    """初始化 ArgumentParser"""
    import argparse
    import pytorch_lightning as pl
       
    parser = argparse.ArgumentParser(add_help=False)

    # Trainer args
    parser = pl.Trainer.add_argparse_args(parser)
    parser._action_groups[-1].title = "Trainer Args"

    # 自定义 args
    parser.add_argument("--wandb", action="store_true", default=False)
    parser.add_argument("--litmodel_class", type=str, default="TransformerLitModel")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--data_class", type=str, default="KGC")
    parser.add_argument("--chunk", type=str, default="")
    parser.add_argument("--model_class", type=str, default="RobertaUseLabelWord")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--task_name", type=str, default=None)
    parser.add_argument("--pretrain", type=int, default=0, help="Set to 1 for pre-training, 0 for link prediction.")

    # 支持 KGC 数据模块参数
    parser.add_argument("--model_name_or_path", type=str, default="roberta-base")
    parser.add_argument("--data_dir", type=str, default="dataset")
    parser.add_argument("--max_seq_length", type=int, default=256)
    parser.add_argument("--eval_batch_size", type=int, default=8)
    parser.add_argument("--overwrite_cache", action="store_true", default=False)
    # parser.add_argument("--batch_size", type=int, default=8)
    # parser.add_argument("--pretrain", action="store_true", default=False)
    # parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--warm_up_radio",type=float,default=0.1,help="Warmup ratio for linear scheduler (e.g., 0.1 means 10% of training steps are warmup)")

    # 临时解析获取 data/model 类名
    temp_args, _ = parser.parse_known_args()
    data_class = _import_class(f"data.{temp_args.data_class}")
    model_class = _import_class(f"models.{temp_args.model_class}")
    lit_model_class = _import_class(f"lit_models.{temp_args.litmodel_class}")

    # 注册 data/model/lit_model 特有参数
    if hasattr(data_class, "add_to_argparse"):
        data_class.add_to_argparse(parser)
    if hasattr(model_class, "add_to_argparse"):
        model_class.add_to_argparse(parser)
    if hasattr(lit_model_class, "add_to_argparse"):
        lit_model_class.add_to_argparse(parser)

    parser.add_argument("--help", "-h", action="help")
    return parser


def main():
    parser = _setup_parser()
    args = parser.parse_args()
    print(args)

    # 固定随机种子
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    pl.seed_everything(args.seed)

    # 动态导入类
    data_class = _import_class(f"data.{args.data_class}")
    model_class = _import_class(f"models.{args.model_class}")
    lit_model_class = _import_class(f"lit_models.{args.litmodel_class}")

    # ===== 加载预训练模型 =====
    # 文本: 所有实体都有文本 → BERT
    print(f"Loading text model from command line argument: {args.model_name_or_path}")
    text_config = BertConfig.from_pretrained(args.model_name_or_path)
    bert_model = BertModel.from_pretrained(args.model_name_or_path)

    # 图像: 只有 HERB 有图像 → CLIP ViT
    clip_model = CLIPModel.from_pretrained('/home/zry/workspace/mkgformer/MKG/models/clip-vit-base-patch32')
    clip_vit = clip_model.vision_model
    vision_config = clip_model.config.vision_config
    # vision_config.device = 'cpu'

    # ===== 构造任务模型 =====
    model = UnimoKGC(vision_config, text_config, pretrain=args.pretrain)

    # ===== 加载预训练权重到模型 =====
    # 权重加载逻辑应该针对 UnimoKGC 内部的 self.encoder
    encoder_dict = model.encoder.state_dict()
    clip_state_dict = clip_vit.state_dict()
    text_state_dict = bert_model.state_dict()

    # 加载 CLIP (Vision) 权重
    for name in encoder_dict:
        if 'vision' in name:
            clip_name = name.replace('vision_', '').replace('embeddings.', 'patch_embedding.')
            if clip_name in clip_state_dict:
                encoder_dict[name] = clip_state_dict[clip_name]

    # 加载 BERT (Text) 权重
    for name in encoder_dict:
        if 'text' in name:
            text_name = name.replace('text_', '')
            if text_name in text_state_dict:
                encoder_dict[name] = text_state_dict[text_name]

    model.encoder.load_state_dict(encoder_dict, strict=False)
    print("✅ Pre-trained weights for Vision (CLIP) and Text (BERT) loaded into UnimoKGC encoder successfully.")

    # ===== 数据 =====
    data = data_class(args, model)
    tokenizer = getattr(data, "tokenizer", None)

    # ===== Lightning 模型 =====
    lit_model = lit_model_class(args=args, model=model, tokenizer=tokenizer, data_config=data.get_config())
    if args.checkpoint:
        print(f"INFO: Loading weights from checkpoint: {args.checkpoint}")
        # =================================================================
        # ✅ 核心修改: 使用 strict=False 来加载权重
        # =================================================================
        checkpoint_state = torch.load(args.checkpoint, map_location="cpu")["state_dict"]
        lit_model.load_state_dict(checkpoint_state, strict=False)
        print("INFO: Checkpoint loaded successfully (strict=False).")

    # Logger & Callbacks
    # =================================================================
    # 1. 配置 Logger
    logger = pl.loggers.TensorBoardLogger("training/logs")
    if args.wandb:
        logger = pl.loggers.WandbLogger(project="kgc_herb", name=args.data_dir.split("/")[-1])
        logger.log_hyperparams(vars(args))

    # 2. 根据模式 (预训练/微调) 动态配置 EarlyStopping 和 ModelCheckpoint
    #    我们不再手动创建 ProgressBar
    if args.pretrain:
        # --- 预训练模式 ---
        early_stop = pl.callbacks.EarlyStopping(monitor="Pretrain/loss", mode="min", patience=3)
        
        checkpoint_cb = pl.callbacks.ModelCheckpoint(
            monitor="Pretrain/loss",
            mode="min",
            filename=f"{args.data_dir.split('/')[-1]}/pretrain-{{epoch}}-{{Pretrain/loss:.2f}}",
            dirpath="output",
            save_weights_only=True
        )
        print("INFO: Callbacks configured for pre-training mode (monitoring Pretrain/loss).")
    else:
        # --- 链接预测模式 ---
        early_stop = pl.callbacks.EarlyStopping(monitor="Eval/mrr", mode="max", patience=5)
        
        checkpoint_cb = pl.callbacks.ModelCheckpoint(
            monitor="Eval/hits10",
            mode="max",
            filename=f"{args.data_dir.split('/')[-1]}/finetune-{{epoch}}-{{Eval/hits10:.2f}}-{{Eval/hits1:.2f}}",
            dirpath="output",
            save_weights_only=True
        )
        print("INFO: Callbacks configured for link prediction mode (monitoring Eval/mrr and Eval/hits10).")

    # 3. 将 Callback 对象组合成一个列表 (不包含 ProgressBar)
    callbacks = [early_stop, checkpoint_cb]   

    # =================================================================
    # Trainer
    # =================================================================
    trainer = pl.Trainer.from_argparse_args(
        args, 
        callbacks=callbacks,
        logger=logger, 
        default_root_dir="training/logs"
    )

    # ===== 训练 & 测试 =====
    if "EntityEmbedding" not in lit_model.__class__.__name__:
        trainer.fit(lit_model, datamodule=data)
        path = checkpoint_cb.best_model_path
        ckpt = torch.load(path, map_location="cpu")
        state_dict = ckpt["state_dict"] if "state_dict" in ckpt else ckpt
        lit_model.load_state_dict(state_dict)

    result = trainer.test(lit_model, datamodule=data)
    print(result)

    if "EntityEmbedding" not in lit_model.__class__.__name__:
        print("*path*"*10)
        print(path)


if __name__ == "__main__":
    main()
