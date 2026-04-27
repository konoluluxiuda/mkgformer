# MKG 项目说明

本目录是论文项目中的核心实验区，围绕 `NEWHERB` 数据集开展两类任务：

1. `recommand/train.py`：中药推荐主任务，也是当前最核心的训练入口。
2. `main.py`：知识图谱补全（KGC）与多模态编码实验入口。

如果你现在是为了接手论文主实验，建议优先阅读：

`recommand/train.py` -> `recommand/model.py` -> `recommand/dataset.py` -> `recommand/config.py` -> `recommand/preprocess_*.py`

## 1. 项目定位

`MKG` 的整体思路可以理解为：

- 先构建一个面向中药领域的多关系知识图谱；
- 再将图结构、草药属性、化学信息、疾病文本等多模态特征注入图神经网络；
- 最终在推荐任务上学习 “疾病 -> 草药集合” 的排序能力。

这里的推荐任务不是普通商品推荐，而是把疾病视作查询端，把草药视作候选端，目标是在候选草药集合里找出最适合该疾病的处方药材。

## 2. 目录结构

`MKG/` 下与论文主线最相关的内容如下：

```text
MKG/
├── dataset/NEWHERB/              # 论文主数据集
│   ├── entities/                 # 各类实体表
│   ├── relation/                 # 原始关系表
│   ├── features/                 # 文本、SMILES、指纹等特征源文件
│   ├── kge_data/                 # KGC 使用的三元组数据
│   ├── recommendation_data/      # 默认推荐图数据
│   ├── paper_graph_data/         # 论文图结构版本
│   ├── tfidf_graph_data/         # TF-IDF Anti-Hub 图结构版本
│   ├── semantic_data/            # 语义相似图版本
│   └── full_graph_data/          # 不做 Top-K 裁剪的图版本
├── recommand/
│   ├── train.py                  # 推荐主训练脚本
│   ├── model.py                  # HMC_GNN_SSL 模型
│   ├── dataset.py                # 图数据加载与采样
│   ├── config.py                 # 推荐任务超参数
│   ├── utils.py                  # 随机种子与评估器
│   ├── preprocess_kge.py         # 默认推荐图构建
│   ├── preprocess_paper_graph.py # 论文图构建
│   └── preprocess_semantic_graph.py
├── models/                       # BERT / CLIP / ChemBERTa / UnimoKGC
├── lit_models/                   # Lightning 封装
├── main.py                       # KGC 训练入口
├── extract_features.py           # 旧版特征提取
├── fuse_features.py              # 文本 + SMILES 融合特征提取
└── scripts/                      # KGC 训练脚本
```

## 3. 主任务：`recommand/train.py`

### 3.1 脚本职责

`recommand/train.py` 是当前推荐实验的统一入口，负责：

1. 选择图结构版本；
2. 加载草药属性、化学特征、疾病文本特征；
3. 构建 `HerbRecDataset` 和 `DataLoader`；
4. 初始化 `HMC_GNN_SSL` 模型；
5. 用 `BPR + 图对比学习 + 跨模态对齐` 进行训练；
6. 在验证集上早停，并在测试集上输出 `Precision/Recall/F1@K`。

### 3.2 训练脚本中的几个关键开关

脚本开头定义了一组实验开关，用来控制论文中的不同变体：

- 图结构选择
  - `USE_PAPER_GRAPH=True`：默认论文图
  - `USE_TFIDF_GRAPH=True`：TF-IDF Anti-Hub 图
  - `USE_FULL_GRAPH=True`：不过滤 Top-K 的全图
  - `USE_SEMANTIC_GRAPH=True`：基于文本相似度的语义图
- 特征注入
  - `USE_BASE_ATTR`：是否使用性味归经等基础属性
  - `USE_CHEM_DENSE`：是否使用化学相关稠密特征
  - `USE_CHEM_FINGERPRINT`：是否拼接化学指纹
  - `USE_DISEASE_TEXT`：是否加入疾病文本特征
- 融合与自监督
  - `FUSION_MODE='gated'`：使用门控融合结构特征/属性特征/化学特征
  - `USE_CROSS_MODAL`：图表示与化学表示做跨模态对比
  - `USE_PROP_CHEM_ALIGN`：属性表示与化学表示做对齐

这部分相当于整篇论文实验设置的总控制台。

### 3.3 数据流

训练流程可以概括为：

```text
图数据(rec_data.pt / edge_index.pt / edge_type.pt)
    + 属性矩阵(node_attributes.pt)
    + 化学矩阵(node_chem_dense.pt / node_chem_fingerprint.pt / node_chem_multihot.pt)
    + 疾病文本矩阵(node_disease_text.pt)
        -> HMC_GNN_SSL
        -> 生成节点表示
        -> BPR 排序学习
        -> 验证集早停
        -> 测试集 Top-K 指标
```

### 3.4 训练目标

`train.py` 里的总损失由四部分组成：

1. `bpr_loss`
   疾病节点与正样本草药的内积得分应高于负样本草药。
2. `graph_ssl_loss`
   对同一张图做两次扰动编码，让相同节点在两种视图中的表示更接近。
3. `cm_ssl_loss`
   让草药节点的图表示与其化学模态表示对齐。
4. `pc_ssl_loss`
   让草药属性表示与化学表示在共享空间中更接近。

总损失形式为：

```text
Loss = BPR + ssl_reg * GraphSSL + CROSS_MODAL_WEIGHT * CrossModal + PROP_CHEM_WEIGHT * PropChem
```

### 3.5 训练与评估策略

- `rec_data.pt` 里先保存了按疾病划分的 `train_dict/test_dict`；
- `train.py` 读取后，会把原始 `test_dict` 再拆成 `val_dict + test_dict`；
- 验证阶段默认关注 `F1@10`；
- 如果连续若干次评估没有提升，就触发 early stopping；
- 最终输出 `P/R/F1@5,10,20,50`。

这意味着论文主结果更依赖推荐排序指标，而不是传统分类准确率。

## 4. 关键代码模块

### 4.1 `recommand/config.py`

这里集中管理推荐任务超参数，包括：

- 数据根目录 `dataset/NEWHERB`
- 默认读取目录 `recommendation_data`
- 训练轮数、学习率、batch size
- 嵌入维度、隐藏维度、dropout
- 对比学习温度与正则项
- `top_k = [5, 10, 20, 50]`

如果只是调训练参数，先看这个文件。

### 4.2 `recommand/dataset.py`

包含两个核心类：

- `GraphDataManager`
  负责读取 `rec_data.pt`、`edge_index.pt`、`edge_type.pt` 和属性矩阵。
- `HerbRecDataset`
  负责为每个疾病采样 `(disease, positive_herb, negative_herb)`。

这里实现的是非常标准的隐式反馈排序训练范式。

### 4.3 `recommand/model.py`

核心模型类是 `HMC_GNN_SSL`，其结构可以概括为：

1. 节点 ID 随机初始化嵌入；
2. 将属性、化学、疾病文本投影到统一维度；
3. 使用加和或门控方式进行模态融合；
4. 经过 `fusion_mlp` 做一次非线性映射；
5. 通过两层 `RGCNConv` 传播；
6. 对两层输出做拼接再融合，得到最终节点表示。

它同时内置了三类自监督能力：

- 图内双视图对比 `calc_ssl_loss`
- 图表示和化学表示对齐 `calc_cross_modal_loss`
- 属性表示和化学表示对齐 `calc_property_chem_loss`

所以这个模型并不是单纯的 GNN，而是 “多模态融合 + 自监督对齐” 的推荐模型。

### 4.4 `recommand/utils.py`

`Evaluator` 的评估逻辑很重要：

- 先得到所有节点嵌入；
- 取疾病节点表示与所有候选草药做内积；
- 按分数排序后计算 `Precision@K / Recall@K / F1@K`。

训练时的 BPR 损失和测试时的内积评分保持一致，这一点保证了训练目标和评估目标对齐。

## 5. 图数据是怎么来的

### 5.1 默认推荐图：`preprocess_kge.py`

该脚本会从 `dataset/NEWHERB/kge_data/*.tsv` 读取三元组，保留核心关系：

- `treats_disease`
- `has_component`
- `has_effect`
- `has_property`
- `belongs_to_meridian`

然后构建：

- 基础异构图边；
- Herb-Herb 协作边；
- Disease-Disease 协作边；
- 推荐任务的 `train_dict/test_dict`。

输出到：

- `dataset/NEWHERB/recommendation_data/edge_index.pt`
- `dataset/NEWHERB/recommendation_data/edge_type.pt`
- `dataset/NEWHERB/recommendation_data/rec_data.pt`

### 5.2 论文图：`preprocess_paper_graph.py`

这个脚本更接近论文中的正式设定：

- 使用 Jaccard 相似度；
- 对 Herb-Herb 和 Disease-Disease 都做阈值过滤；
- 再做 Top-K 保留，抑制高频枢纽节点的干扰。

输出目录是 `paper_graph_data/`，也是 `train.py` 当前默认使用的版本。

### 5.3 语义图：`preprocess_semantic_graph.py`

该脚本使用实体文本描述做：

- TF-IDF 编码；
- 余弦相似度计算；
- Top-K 语义邻居构图。

适合做 “结构图 vs 语义图” 的对比实验。

## 6. 特征文件说明

`train.py` 会直接读取下列文件：

- `node_attributes.pt`
  草药属性特征，主要是性味归经等 multi-hot 信息。
- `node_chem_dense.pt`
  化学稠密表示，用于跨模态对齐。
- `node_chem_fingerprint.pt`
  化学指纹，可与稠密表示拼接。
- `node_chem_multihot.pt`
  化学多热属性特征，作为注入特征的一部分。
- `node_disease_text.pt`
  疾病文本表示，用于缓解疾病冷启动。

这些文件目前已经存在于多个图目录中，例如：

- `recommendation_data/`
- `paper_graph_data/`
- `tfidf_graph_data/`
- `full_graph_data/`

也就是说，切换图结构时，训练脚本不仅切换边，还会连带切换同目录下的特征文件。

## 7. `main.py` 在整个项目中的位置

`main.py` 不是推荐任务入口，而是多模态知识图谱补全入口，基于 PyTorch Lightning 组织训练。它的职责是：

- 动态加载 `data.*`、`models.*`、`lit_models.*`；
- 初始化文本编码器 BERT 和视觉编码器 CLIP；
- 构建 `UnimoKGC`；
- 执行预训练或链接预测微调。

如果论文某一部分涉及 “MKG 预训练表示” 或 “多模态实体编码”，那条线主要看：

`main.py` -> `models/model.py` -> `lit_models/`

如果只是复现最终推荐结果，则可以先不深入这部分。

## 8. 推荐的阅读顺序

建议按下面顺序理解项目：

1. 看 `recommand/train.py`
   明白实验开关、训练目标、验证和测试流程。
2. 看 `recommand/model.py`
   明白模型由哪些模态组成，损失是怎么拼起来的。
3. 看 `recommand/dataset.py`
   明白训练样本和图结构是如何加载的。
4. 看 `recommand/preprocess_paper_graph.py`
   明白论文图是如何构造出来的。
5. 再回看 `dataset/NEWHERB`
   对照真实数据文件理解每个输入矩阵来源。
6. 最后看 `main.py`
   补上 KGC 和多模态编码的上游背景。

## 9. 运行方式

### 9.1 运行推荐主任务

在 `MKG/recommand/` 下执行：

```bash
python train.py
```

脚本本身已经在开头写死了实验开关，因此当前运行方式更偏向“改代码式实验”，而不是命令行参数式实验。

### 9.2 重建默认推荐图

```bash
cd MKG/recommand
python preprocess_kge.py
```

### 9.3 重建论文图

```bash
cd MKG/recommand
python preprocess_paper_graph.py
```

### 9.4 重建语义图

```bash
cd MKG/recommand
python preprocess_semantic_graph.py
```

### 9.5 KGC 训练

仓库里已经给了脚本示例：

```bash
cd MKG
bash scripts/train_herb_kgc.sh
```

## 10. 当前项目的几个注意点

1. `MKG` 下不存在顶层 `train.py`，推荐主入口实际是 `recommand/train.py`。
2. `train.py` 中图结构开关是互斥思路，建议一次只保留一个为 `True`。
3. 特征文件路径目前部分写死为 `recommendation_data`，而属性路径来自 `Config.REC_DATA_DIR`，做图切换实验时需要特别留意是否完全同步。
4. `requirements.txt` 里的环境版本偏老，尤其是 `torch`、`pytorch_lightning`、`transformers` 和 `torch_geometric`，复现时尽量保持兼容版本。
5. 根目录还有 `BSGAM`、`PTM`、`PresRecRF`、`TCMPR` 等基线模型，说明这个仓库不仅包含主模型，也承担论文对比实验。

## 11. 一句话总结

如果只抓主线，这个项目可以概括为：

“基于 `NEWHERB` 异构知识图谱，将结构信息、草药属性、化学特征和疾病文本融合进 `HMC_GNN_SSL`，并通过排序损失与多种自监督对齐策略完成中药推荐实验。”
