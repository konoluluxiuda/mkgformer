#!/bin/bash

# =================================================================
# Stage 2: Fine-tuning for Knowledge Graph Completion on HERB Dataset
# =================================================================

python main.py \
  --gpus 1 \
  --max_epochs 20 \
  --num_workers 4 \
  --accumulate_grad_batches 2 \
  \
  --model_name_or_path models/bert-base-chinese \
  --model_class UnimoKGC \
  \
  `# === 核心修改：切换到微调模式 ===` \
  --pretrain 0 \
  --checkpoint /home/zry/workspace/mkgformer/MKG/output/HERB/pretrain-epoch=11-Pretrain/loss=2.10.ckpt\
  --lr 5e-5 \
  \
  `# === 数据与任务设置 ===` \
  --data_dir dataset/HERB \
  --task_name HERB \
  --batch_size 64 \
  --eval_batch_size 96 \
  --max_seq_length 128 \
  \
  `# === 其他设置 ===` \
  --check_val_every_n_epoch 1 \
  --overwrite_cache \
  --warm_up_radio 0.1