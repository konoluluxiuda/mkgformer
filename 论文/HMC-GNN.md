HMC-GNN: Multimodal Knowledge-Enhanced Heterogeneous Graph Learning for Disease-Driven Traditional Chinese Medicine Prescription Recommendation

## Abstract

Traditional Chinese Medicine (TCM) prescription recommendation requires modeling complex and synergistic herb combinations. Although graph-based methods have advanced automated herb recommendation, they still face two major challenges in disease-driven settings. First, the scale-free nature of TCM graphs often leads to hub-node bias, where highly frequent herbs form overly dense connections that aggravate over-smoothing and weaken long-tail therapeutic signals. Second, directly combining high-dimensional molecular semantics with sparse topological or pharmacological attributes may result in suboptimal cross-modal alignment, limiting the effective integration of heterogeneous knowledge. To address these issues, we propose HMC-GNN, a heterogeneous multimodal contrastive graph neural network for disease-driven TCM prescription recommendation. Specifically, we first construct an asymmetric collaborative graph based on Jaccard similarity and Top-K pruning to reduce popularity bias while preserving informative synergistic neighborhoods. We then introduce an adaptive gated multimodal fusion module to align and integrate macroscopic TCM properties with microscopic chemical descriptors, including molecular fingerprints and dense chemical representations. On this basis, an RGCN-based encoder is trained with a joint objective that combines collaborative ranking and cross-modal self-supervised learning, encouraging structurally meaningful and semantically aligned node representations. Extensive experiments on a multimodal TCM benchmark show that HMC-GNN consistently outperforms competitive baselines such as KDHR, BSGAM, and PresRecRF across major evaluation metrics. In addition, geometric visualization results provide qualitative evidence that the proposed framework learns more structured and interpretable representations, supporting intelligent disease-driven herb recommendation.

Keywords: Traditional Chinese Medicine, Disease-Driven Prescription Recommendation, Heterogeneous Graph Neural Network, Multimodal Fusion, Contrastive Learning

## Introduction

Traditional Chinese Medicine (TCM) has evolved over thousands of years into a systematic and holistic medical framework \[1-3\]. Unlike modern Western medicine, which often targets specific molecular pathways using single active compounds \[4\], TCM treatment commonly relies on herbal prescriptions composed of multiple herbs with complementary or synergistic effects. The formulation of these prescriptions follows the classic compatibility principle of “Monarch, Minister, Assistant, and Courier” (Jun-Chen-Zuo-Shi), in which different herbs cooperate to address complex pathological conditions from multiple perspectives \[5-7\]. With the growing demand for the modernization and digitalization of TCM, Artificial Intelligence (AI) has been increasingly introduced to analyze prescription patterns and support intelligent recommendation \[8-10\].

Most existing AI studies on TCM prescription recommendation focus on **symptom-driven** settings, where a relatively rich set of symptoms is used to predict candidate herbs \[11,12\]. However, this setting does not fully reflect an increasingly important scenario in modern integrative medicine, namely the paradigm of **integrating disease diagnosis with syndrome differentiation** \[13-15\]. In real clinical practice, patients are often first diagnosed with a specific modern disease, such as hypothyroidism or type 2 diabetes, before receiving further TCM consultation. This gives rise to the task of **disease-driven prescription recommendation**, whose goal is to generate a core herbal formula directly from a disease diagnosis \[16,17\]. Compared with symptom-driven recommendation, this task is more challenging because the input is much sparser: instead of a dense symptom set, the model often receives only a single disease node. As a result, the recommender must rely more heavily on external knowledge and multimodal semantics to infer appropriate herb combinations \[18-20\].

Existing computational methods for TCM prescription recommendation can be roughly grouped into three categories. The first category includes **statistical and topic-modeling approaches**, which treat prescriptions as documents and herbs or symptoms as words to mine latent therapeutic patterns \[21,22\]. Representative methods such as PTM \[41\], SHDT \[42\], and MCLDA \[23\] model latent associations among symptoms, pathogenesis, and herbs, while KGETM \[43\] further incorporates knowledge graph embedding into the topic modeling framework. These methods provide useful probabilistic insights, but their representation ability is often limited when modeling fine-grained structural dependencies and higher-order pharmacological interactions.

The second category consists of **sequence-based generative models**, which formulate prescription generation as a sequence-to-sequence task \[24,25\]. Models such as TCM Translator \[44\], AttentiveHerb \[45\], Herb-Know \[46\], and TCMBERT \[47\] use recurrent or Transformer-based architectures to map symptom inputs to herb sequences, sometimes with the help of external textual knowledge. More recent methods, including PreGenerator \[48\] and GSCCAM \[49\], further introduce compatibility-aware or dual-branch mechanisms to improve generation quality. Although these models are effective in some settings, representing prescriptions as ordered sequences may weaken the modeling of the inherently unordered and synergistic nature of herbal combinations.

The third category, and currently the most relevant to this work, is **knowledge-graph-based graph neural networks (GNNs)** \[26,27\]. By modeling TCM entities and their relations in graph form, these methods can explicitly capture structured therapeutic associations and have shown clear advantages over earlier statistical or sequential models. Representative studies such as SMGCN \[50\] and KDHR \[28\] demonstrate that graph-based propagation can improve herb recommendation by leveraging symptom-herb or formula-herb relations. Subsequent methods, including MGCN \[51\], TCM-GCN \[52\], and SMRGAT \[53\], further enhance representation learning through multi-layer aggregation and attention mechanisms. These advances indicate that graph learning provides a promising foundation for prescription recommendation.

However, despite their effectiveness, existing graph-based methods still face important limitations when applied to the **disease-driven** setting. The first limitation lies in **hub-node bias and topological over-smoothing**. TCM data typically exhibit a long-tailed distribution, in which a small number of highly frequent herbs, such as Licorice or Ginseng, appear in a large number of prescriptions \[30\]. When collaborative graphs are constructed directly from co-occurrence statistics, these ubiquitous herbs tend to become dominant hub nodes, producing overly dense and noisy connections. During graph propagation, such structures may aggravate over-smoothing and weaken the model’s ability to preserve discriminative signals for specific or rare diseases \[31,32\].

The second limitation concerns **cross-modal misalignment between structural and semantic information**. To compensate for sparse disease inputs, recent studies have begun to incorporate dense multimodal features, such as textual embeddings, SMILES-based molecular representations, and chemical descriptors \[33,34\]. While these features can enrich node semantics, simple direct concatenation may cause high-dimensional dense features to dominate sparse topological signals, resulting in suboptimal multimodal integration \[35\]. More importantly, many existing methods still lack an explicit mechanism to align **macroscopic TCM knowledge** (e.g., medicinal properties and meridian tropisms) with **microscopic biochemical information** (e.g., molecular fingerprints and chemical semantics) in a shared representation space \[36,37\]. This weakens the model’s ability to jointly exploit traditional pharmacological theory and modern molecular evidence.

Motivated by these challenges, we propose **HMC-GNN**, a **Heterogeneous Multimodal Contrastive Graph Neural Network** for disease-driven TCM prescription recommendation. The proposed framework is designed around three key ideas. First, to reduce popularity bias while preserving informative synergistic neighborhoods, we construct an asymmetric collaborative graph using **Jaccard-based Top-K pruning**. Second, to better integrate heterogeneous domain knowledge, we introduce an **adaptive gated multimodal fusion** module that aligns and combines structural embeddings, TCM attribute features, and chemical descriptors. Third, to improve representation learning under sparse disease inputs, we optimize the model with a **multi-view self-supervised objective** that combines graph-level contrastive learning and cross-modal alignment. Through these components, HMC-GNN aims to learn node representations that are both structurally informative and semantically aligned.

To evaluate this problem systematically, we build **ETCM-HERBKG**, a multimodal benchmark derived from the authoritative ETCM database \[38\]. Experimental results on this benchmark show that HMC-GNN consistently outperforms several competitive baselines, including KDHR, BSGAM, and PresRecRF, across major evaluation metrics. In addition, representation visualization provides qualitative evidence that the learned latent space captures meaningful pharmacological structure and improves multimodal interpretability.

The main contributions of this work are summarized as follows:

- We propose an asymmetric collaborative graph construction strategy based on Jaccard similarity and Top-K pruning to alleviate hub-node bias and reduce over-smoothing in disease-driven TCM prescription recommendation.
- We design an adaptive gated multimodal fusion framework to align and integrate structural embeddings, macroscopic TCM attributes, and microscopic chemical descriptors in a shared latent space.
- We develop a joint self-supervised optimization framework that combines collaborative ranking, structural contrastive learning, and cross-modal alignment. Extensive experiments on the ETCM-HERBKG benchmark demonstrate that HMC-GNN achieves strong performance compared with competitive baselines while yielding more interpretable latent representations.

## Problem Formulation

In this section, we formally define the notations utilized throughout this paper and formulate the intelligent disease-driven herb recommendation task as an implicit feedback ranking problem on a multimodal heterogeneous graph.

### Notations and Definitions

For clarity, the key mathematical notations used throughout this paper are summarized in Table 1. Let  and  denote the sets of diseases (or clinical syndromes) and candidate herbs, respectively, where  and  represent their total numbers.

To capture complex therapeutic interactions and rich physical modalities, we construct a unified multi-relational heterogeneous graph, defined as . The node set  comprises not only the core diseases and herbs (), but also multi-granularity attribute nodes, denoted as  (e.g., macroscopic TCM properties and microscopic chemical fingerprints). Crucially, the multi-relational edge set  encompasses various interaction types  (e.g., treats_disease, has_property, and collaborative_co_occurrence). This multi-relational topological nature serves as the fundamental theoretical prerequisite for applying Relational Graph Convolutional Networks (RGCN), explicitly empowering the model to distinguish and aggregate information across entirely different semantic pathways during the message-passing phase.

Table 1.

Nomenclature

|     |     |
| --- | --- |
| Notation | Description |
|     | Set of diseases (syndromes) and candidate herbs |
|     | Total number of diseases and candidate herbs |
|     | The unified multi-relational heterogeneous TCM graph |
|     | Sets of nodes, edges, and relation types in |
|     | Set of multi-granularity attribute nodes (TCM properties, chemical fingerprints) |
|     | A specific relation type (e.g., treats, has_property) |
|     | An observed positive herb and an unobserved negative herb for a disease |
|     | The \-dimensional latent representations of disease  and herb |
|     | The predicted representation-based ranking score between  and |

### Task Formulation

We formulate the prescription generation as an implicit feedback ranking task. Under the implicit feedback setting, the clinical dataset only provides observed positive interactions, indicating that an herb  is historically prescribed for a specific disease condition .

Our ultimate goal is to learn a robust representation-based scoring function , where  and  respectively denote the final \-dimensional latent representations of the disease and the herb learned by our model, and  denotes the inner product. The objective of the recommendation system is to optimize these representations such that the score of an observed positive herb  is consistently ranked higher than that of an unobserved negative herb  sampled from the candidate space . By successfully modeling this ranking boundary, the model can recommend a highly precise and pharmacologically sound subset of herbs for any given complex disease query.

## Proposed Methodology

### Overall Architecture

The overall architecture of the proposed HMC-GNN framework is illustrated in **Fig 1**. To bridge the semantic chasm between high-dimensional molecular chemistry and macroscopic TCM theories, our model is explicitly designed with a four-stage pipeline:

- **Data Input & Robust Graph Construction (Section 4.2):** We first construct a heterogeneous disease-herb bipartite network, employing an asymmetric Top-K Jaccard pruning strategy to actively filter generic hub-node noise and reconstruct a high-quality sparse topology.
- **Multimodal Feature Alignment and Gated Fusion (Section 4.3):** We initialize the nodes using three disparate semantic spaces: topological structural embeddings, macroscopic TCM attributes (properties and meridian tropisms), and microscopic chemical descriptors (dense vectors and molecular fingerprints). Following initial linear alignment, an adaptive sigmoid-gated mechanism dynamically fuses these heterogeneous modalities into unified representations.
- **Unified Heterogeneous Graph Reasoning (Section 4.4):** The effectively fused node embeddings are fed into a multi-layer explicit message-passing neural network to capture high-order synergistic compatibilities across the reconstructed graph.
- **Quadruple Joint Optimization Framework (Section 4.5):** To prevent topological over-smoothing and modality overshadowing, we jointly optimize the primary BPR recommendation loss () alongside three auxiliary self-supervised learning (SSL) penalties: intra-graph structural contrastive loss (), cross-modal distillation loss ), and Property-Chemical semantic alignment loss ().

Fig. 1. The overall architecture of the proposed HMC-GNN.

### Robust Graph Construction via Top-K Jaccard Pruning

A primary challenge in TCM datasets is the severe popularity bias (also known as the hub-node problem), inherently evidenced by the classic power-law co-occurrence distribution (**Fig. 2c**). A few ubiquitous "harmonizing" herbs (e.g., Ginseng or Licorice) participate in hundreds of prescriptions, creating a misleading "panacea" illusion. Consequently, constructing collaborative graphs based purely on raw co-occurrence frequencies inevitably generates over-dense, indiscriminate topologies (**Fig. 2a**). Such hairball-like structures not only exacerbate the over-smoothing issue in Graph Neural Networks (GNNs) but also severely drown out the specific therapeutic signals of long-tail herbs.

To address this bottleneck, we design a robust asymmetric collaborative graph construction module leveraging Jaccard normalization and Top-K Jaccard Pruning to uncover genuine synergistic patterns. Specifically, the collaborative similarity between two herbs and is evaluated based on the overlap of their treated diseases, formulated as Eq. (1):

where denotes the set of diseases associated with herb . Crucially, **the union term in the denominator inherently acts as a robust penalty for massive hub nodes**. For ubiquitous herbs with extremely large degrees, the expanded denominator effectively diminishes their similarity scores with ordinary herbs, mathematically stripping away their spurious "universal synergistic" illusions.

To further denoise the topology and maintain a highly informative, sparse structure, we enforce a strict Top-K Jaccard Pruning over the similarities derived from Eq. (1). The tailored collaborative neighborhood for each herb is formulated as Eq. (2):

where is a strictly pre-defined similarity threshold. This truncation operation intrinsically yields an **asymmetric directed graph**: herb being in the Top-K relevant neighborhood of does not guarantee the reciprocal.

As visual proof of this theoretical design, **Fig. 2b** vividly demonstrates the transformation: the massive hub node (Ginseng) is effectively penalized and mathematically detached (isolated) due to failing the rigorous threshold requirement. Concurrently, the true synergistic pairings among long-tail herbs—which share a high density of local targets—successfully survive the pruning to form clear, closed-loop local clusters. With the outbound degrees strictly bounded to an upper limit of (**Fig. 2d**), this strategy thoroughly purges low-confidence correlations and naturally restores a denoised, highly discriminative collaborative structure for unbiased representation learning.

Fig. 2. Illustration of robust collaborative graph construction via Top-K Jaccard pruning.

The raw ego-network sampled around the popular hub herb (Ginseng), showing over-dense co-occurrence connections. (b) The pruned asymmetric ego-network generated by applying Jaccard normalization and Top-K Jaccard Pruning (). (c) The classic power-law distribution of co-occurrence frequencies between herbs in the dataset. (d) The out-degree distribution of the nodes in the pruned subgraph, tightly constrained by the pre-defined capacity limit ().

### Explicit Multimodal Feature Alignment and Fusion

To equip the graph nodes with rich domain knowledge, we extract three distinct multimodal features: learnable structural embeddings , macroscopic TCM attributes (multi-hot property/meridian vectors) , and microscopic chemical features (ChemBERTa dense vectors concatenated with molecular fingerprints) . However, directly concatenating these heterogeneous modalities often leads to dimensional imbalance and semantic collision.

To align these initial features, we first project the high-dimensional sparse and dense features into a unified -dimensional latent space using modality-specific linear transformations, as defined in Eq. (3) and Eq. (4):

where and denote the trainable weight matrices and bias vectors for each modality, respectively.

Subsequently, we propose an Adaptive Gated Multimodal Fusion mechanism. Recognizing that different herbs rely on varying informative modalities (e.g., some are dominated by their topological roles, while others by specific active chemical compounds), we employ a Softmax-based gating network to adaptively regulate the contribution of each modality at the node level. As shown in Eq. (5), we first compute the modality-specific importance weights:

where denotes the concatenation operation, and represents the learned importance weight for the structural, macroscopic, and microscopic modalities, naturally constraining their sum to (i.e., ). Finally, the adaptive fusion dynamically balances these different semantic signals, yielding a highly expressive and comprehensive initial representation , as derived in Eq. (6):

This adaptively fused embedding is then fed into the subsequent message-passing layers to ensure that structural propagation is continually guided by deep multimodal semantics.

### Unified Heterogeneous Graph Reasoning

The fused initial representation serves as the -th layer node embedding , which is then fed into a unified Relational Graph Convolutional Network (RGCN) to capture high-order collaborative signals across the heterogeneous graph . Unlike standard GCNs, RGCN applies relation-specific weight transformations to preserve the distinct semantic meanings of different edge types (e.g., distinguishing therapeutic efficacy from property assignment). The message-passing process at the -th layer is defined in Eq. (7):

where denotes the neighborhood of node under relation , is a structure-based normalization constant, is the trainable relation-specific transformation matrix, models the self-loop connection, and is the non-linear activation function.

A widely recognized limitation of deep GNNs is the over-smoothing effect, where node representations become indistinguishable as the network depth increases. To circumvent this and capture both local and global topologies, we implement a layer aggregation strategy. Instead of relying solely on the final layer's output, we concatenate the intermediate embeddings from different hops (e.g., Layer and Layer ) and apply a linear projection to obtain the final comprehensive node representation , as formulated in Eq. (8):

Where and are the projection weights and biases, respectively. This residual-like aggregation effectively bridges multi-hop neighborhood information while preserving the specific topology structure of the ego-network, substantially enhancing the discriminative power of the final representations for downstream tasks.

### Quadruple Joint Optimization Framework

To fully exploit the multimodal graph structure and mitigate data sparsity, we design a comprehensive Quadruple Joint Optimization framework. The total objective function is formulated as Eq. (9):

where and are hyperparameters weighting the strengths of the auxiliary self-supervised learning (SSL) signals. Each component plays a highly specialized role in regularizing the representation latent space:

- **Recommendation Loss ():** As the primary task objective, we employ the Bayesian Personalized Ranking (BPR) \[54\] loss to optimize pairwise ranking preferences. It ensures that the algorithm preserves high-order collaborative signals by scoring an observed positive disease-herb pair higher than an unobserved negative one. This pairwise loss is computed as shown in Eq. (10):

where  denotes the training set containing observed positive interactions () along with negative counterparts () via uniform random sampling. Specifically, for each observed positive herb , we sample an unobserved herb  from the candidate set , where  represents the set of herbs known to treat disease . Here,  denotes the predicted pairing score, and  is the sigmoid activation function.

- Intra-Graph SSL (): Through dual-view edge perturbation (e.g., stochastic edge dropout), we generate two augmented topological views. We apply the InfoNCE \[55\] loss to maximize the mutual information of the same node across the two variants.  Let  and  represent the embeddings of node  in the two augmented views. The formula is mathematically expanded as Eq. (11):

where  denotes the cosine similarity function, and  denotes the temperature hyperparameter. Here, the identical node across two views  constitutes the positive pair, while representations of other distinct nodes  within the batch serve as negative samples.

- **Cross-Modal SSL ():** To prevent semantic drift during deep graph propagation, we contrast the GNN-output structural features () with the raw, independently mapped microscopic chemical features (). Utilizing an InfoNCE-based alignment, this objective is defined in Eq. (12):

This objective mathematically anchors the topologically smoothed graph embeddings back to their objective biochemical essence, ensuring structural learning does not override intrinsic molecular properties.

- **Macroscopic-Microscopic Property Alignment ():** Within a shared latent space, we enforce a contrastive constraint to directly pull macroscopic TCM attributes (e.g., Four Natures, Five Flavors, parameterized as ) closer to their corresponding microscopic biochemical structures  (). The alignment loss is computed as Eq. (13):

This auxiliary objective provides a profound modern chemical corroboration for ancient TCM theories. Positive samples are the matched macro-micro properties of the exact same herb, while negative samples are mismatched properties from different herbs, ensuring absolute semantic consistency across inherently heterogeneous modalities.

## Experiment and Results

### Research Questions

To rigorously validate the effectiveness and theoretical soundness of our proposed HMC-GNN framework, we meticulously design our experiments to answer the following critical research questions (RQs):

**RQ1 (Overall Performance):** Does our proposed model outperform state-of-the-art baselines on disease-driven prescription recommendation, particularly in mitigating the severe long-tail distribution?

**RQ2 (Graph Topology & Hub-node Mitigation):** How effectively does the Asymmetric Graph Construction strategy mitigate "popularity bias"? Specifically, does the Top-K semantic thresholding act as an effective information bottleneck to filter out generic noise while preserving localized high-quality topologies?

RQ3 (Multimodal Fusion via SSL): Can the Contrastive Latent Space Alignment (via and ) resolve the "semantic collision" between extreme dimensional spaces? We aim to verify if our dual-view SSL successfully projects discrete TCM properties and continuous chemical vectors into a coherent latent space without mutual interference.

**RQ4 (Hyperparameter Sensitivity):** How resilient is the performance of HMC-GNN to variations in critical hyperparameters, such as the SSL loss weights and the topological truncation threshold ?

**RQ5 (Geometric Interpretability):** Are the learned multimodal embeddings geometrically consistent with historical TCM compatibility theories? We qualitatively investigate whether the latent space inherently reveals pharmacological clusters (e.g., aggregating herbs with similar molecular structures or clinical natures).

To address the above RQs, the remainder of this section is organized as follows. First, we detail the experimental settings, including datasets, evaluation metrics, and baseline configurations. We then present the overall performance comparison (RQ1), followed by detailed ablation studies on graph topology and multimodal fusion (RQ2 & RQ3). Finally, we systematically evaluate the impact of hyperparameters (RQ4) and provide visual case studies to illustrate the geometric interpretability of our model (RQ5).

### Experimental Settings

**Datasets**

To comprehensively evaluate the proposed HMC-GNN model, we conduct experiments on ETCM-HerbKG, a newly constructed multimodal benchmark dataset tailored for Traditional Chinese Medicine (TCM) prescription recommendation. Our task focuses on recommending a customized, synergistic set of herbs given a specific disease query. The detailed statistical information of the dataset, including the number of entities, interactions, and data splits, is summarized in Table 2.

Table 2.

Statistical Information of the ETCM-HerbKG Dataset.

|     |     |     |     |     |     |     |     |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Dataset | Diseases | Herbs | TCM Properties | Total Edges | Train | Val | Test |
| ETCM-HERBKG | 2,703 | 395 | 24  | 591,414 | 198,065 | 24,960 | 25,805 |

The construction of ETCM-HerbKG rigorously integrates knowledge from two authoritative sources:

- Entity and Modality Initialization: The foundational herb entities and their intrinsic macroscopic TCM properties (e.g., the Four Natures, Five Flavors, and meridian tropisms) were systematically curated from the Encyclopedia of Traditional Chinese Medicine (ETCM). To enable robust cross-modal learning, we explicitly extracted microscopic 166-dimensional discrete MACCS molecular fingerprints and continuous multidimensional chemical representations (via pre-trained ChemBERTa embeddings) for these corresponding herbs.
- Disease-Herb Interaction Network: To formulate the bipartite recommendation graph, the clinical disease-herb interaction records were filtered and strictly aligned based on the public KDHR dataset.

Data Distribution Characteristics: A critical statistical attribute of the ETCM-HerbKG dataset is its highly skewed power-law distribution regarding herb occurrence frequencies (as previously illustrated in **Fig. 2c**). Statistical analysis reveals that a scarce few "hub herbs" (e.g., _Glycyrrhiza uralensis_ / Licorice, _Radix Ginseng_ / Ginseng) participate in over 80% of the disease-prescription subgraphs, while a vast majority of niche herbs are sparsely linked. From an evaluation perspective, this extreme popularity bias \[56\] poses a severe challenge, as it typically forces conventional GNN aggregators into severe over-smoothing, heavily favoring hub nodes over long-tail therapeutic signals. As detailed in Section 4.2, this specific topological extremity serves as the underlying data rationale for implementing our Asymmetric Collaborative Graph Construction module.

Baselines

To evaluate the effectiveness of the proposed HMC-GNN framework, we compare it against five representative baseline models. To ensure a fair comparative analysis, baselines originally designed for symptom-driven tasks have been structurally adapted to accommodate our disease-driven recommendation paradigm. The selected baselines span traditional statistical modeling, classical machine learning, and advanced graph-based architectures:

**Traditional Statistical Modeling:**

PTM\[57\]: The Probabilistic Topic Model is a generative approach that regards diseases (originally symptoms) and herbs as words distributed over latent therapeutic topics. It explicitly incorporates TCM domain knowledge into the prescription generation process to model probabilistic co-occurrence patterns .

**Machine Learning & Shallow Representation:**

TCMPR\[58\]: TCMPR is a recommendation method that employs a subnetwork-based mapping approach. It leverages matrix projections to extract substructures from the knowledge network, enabling the representation learning of disease entities and herbs based on existing connectivity relationships.

PresRecRF \[25\]: This method relies on classical machine learning paradigms for prescription recommendation. It constructs dense feature sets based on entity attributes and utilizes Random Forest ensembles to learn the mapping from diagnostic inputs to target herbs.

**Graph-based Knowledge-aware Models:**

KDHR \[28\] (Knowledge-aware Disease-Herb Recommendation): KDHR constructs an explicit TCM knowledge graph and applies a multi-layer information fusion mechanism. It leverages predefined meta-paths (e.g., Disease-Symptom-Herb) to aggregate neighborhood information, thereby integrating hierarchical graph structure and node attributes to uncover complex network correlations.

BSGAM \[29\] (Bipartite Sub-Graph Attention Model): BSGAM is a recent attention-based GNN architecture tailored for bipartite recommendation tasks. It processes large-scale macroscopic networks by decomposing them into smaller, localized subgraphs, upon which attention mechanisms are applied to aggregate local structural information and compute herb recommendation scores.

Our proposed HMC-GNN introduces a different paradigm compared to the aforementioned methods. Instead of relying on predefined meta-paths or localized subgraph partitioning, it employs an end-to-end multimodal graph contrastive learning framework designed to preserve global topological integrity while directly fusing deep textual and chemical spatial features (e.g., ChemBERTa semantics and MACCS fingerprints).

**Evaluation Metrics**

To comprehensively evaluate the performance of our disease-driven herb recommendation system, we adopt three widely recognized metrics: Precision@, Recall@, and F1-Score@, as mathematically illustrated in Eqs. (11), (12), (13). Given a specific disease query , let denote the set of Top-K herbs recommended by our model, and denote the ground-truth set of herbs historically prescribed for that disease. The metrics for a single query are formally defined as follows:

Where Eq. (11) represents precision, which quantifies the proportion of correctly recommended herbs among the top-K outputs. Eq. (12) represents recall, which measures the proportion of ground-truth herbs successfully captured by the recommendation list. Eq. (13) is the harmonic mean of precision and recall, providing a comprehensive and balanced evaluation of both exactness and completeness.

In our experiments, we report the average values of these metrics across all tested diseases , setting the cutoff length .

### Overall Performance (RQ1)

As presented in Table 3, the proposed HMC-GNN framework achieves the highest performance across all evaluation metrics. Utilizing the comprehensive F1-score as the primary benchmark, our model exhibits substantial improvements over the competitive baselines. Specifically, HMC-GNN outperforms TCMPR, KDHR, and BSGAM by **39.5%**, **69.1%**, and **81.4%** in F1@10, respectively. Furthermore, in terms of Recall@10, our approach achieves a score of **0.3435**, compared to **0.1521** for the strongest graph-based baseline (KDHR). These performance margins demonstrate that HMC-GNN effectively balances precision and recall, establishing its superiority in disease-driven prescription generation.

Table 3.

Overall Performance Comparison.

|     |     |     |     |     |     |     |
| --- | --- | --- | --- | --- | --- | --- |
| **Model** | **P@5** | **R@5** | **F1@5** | **P@10** | **R@10** | **F1@10** |
| PTM-A | 0.1173 | 0.0710 | 0.0632 | 0.1122 | 0.1187 | 0.0831 |
| TCMPR | 0.1348 | 0.0848 | 0.0736 | 0.1250 | 0.1360 | 0.0953 |
| PresRecRF | 0.1283 | 0.0704 | 0.0679 | 0.1239 | 0.1367 | 0.0935 |
| KDHR | 0.1255 | 0.0903 | 0.0768 | 0.1232 | 0.1521 | 0.1035 |
| BSGAM | 0.1356 | 0.0858 | 0.0747 | 0.1273 | 0.1391 | 0.0965 |
| **HMC-GNN (Ours)** | **0.2029** | **0.2373** | **0.1557** | **0.1853** | **0.3435** | **0.1751** |
| Improve. By TCMPR | 50.52% | 179.83% | 111.55% | 48.24% | 152.57% | 83.74% |
| Improve. By KDHR | 61.67% | 162.79% | 102.73% | 50.41% | 125.84% | 69.18% |
| Improve. By BSGAM | 49.63% | 176.57% | 108.43% | 45.56% | 146.94% | 81.45% |

Comparative analysis reveals the underlying reasons for the performance differences among the baselines. Traditional statistical and shallow representation models (e.g., PTM-A, TCMPR, and PresRecRF) rely heavily on direct co-occurrence statistics or subnetwork projections. Due to the extreme sparsity of direct disease-herb interactions, many nodes lack sufficient association information, resulting in limited feature updates and sub-optimal overall performance. In contrast, advanced graph-based models (KDHR and BSGAM) achieve improved results. KDHR incorporates external herb attributes and multi-layer information fusion, while BSGAM utilizes graph attention mechanisms over bipartite subgraphs to emphasize specific structural roles. Consequently, these models capture higher-order therapeutic mappings more effectively.

However, the performance of these advanced graph models is still limited by two structural factors. First, they operate on raw, scale-free TCM networks where connectivity is overwhelmingly dominated by a few "hub" herbs (e.g., _Glycyrrhiza uralensis_). This unfiltered propagation often leads to over-smoothing, biasing the recommendations toward generic herbs and neglecting specific, long-tail candidates. Second, their feature integration lacks the depth to bridge the gap between discrete topological structures and continuous micro-molecular properties. HMC-GNN overcomes these limitations by introducing asymmetric Top-K Jaccard Pruning to mitigate hub-node interference and employing multimodal contrastive learning to supplement sparse topological signals with rich chemical-property semantics. Detailed empirical validations of these mechanisms are presented in the subsequent ablation studies (RQ2 & RQ3).

### Ablation Study (RQ2 & RQ3)

To validate the efficacy of the key components in the proposed HMC-GNN framework, we conducted a systematic ablation study. The structural mechanisms and multimodal fusion components evaluated include:

- Topological Information Bottleneck (W/O Top-K): Removes the local Top-K neighborhood filter, retaining unfiltered meta-path connections.
- Unified Heterogeneous Connectivity (W/O Unified): Decouples the unified heterogeneous graph into isolated homogeneous subgraphs (e.g., Symptom-Herb, Herb-Herb).
- Structural Normalization (W/O Jaccard): Replaces the Jaccard frequency thresholding with traditional TF-IDF penalization logic.
- Incorporation of Molecular Knowledge (W/O Chem): Excludes modern chemical representations (e.g., SMILES features).
- Cross-Modal Contrastive Alignment (W/O SSL): Disables the cross-modal contrastive losses, replacing them with naïve feature concatenation.

For these components, we systematically eliminated or replaced them to observe their impact on performance. The experimental design and the performance degradation rates (DR) relative to the full model are illustrated in **Table 4**.

Table 4.

Experimental results of the ablation study.

|     |     |     |     |     |     |     |     |     |     |     |     |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| **Method** | **Top-K** | **Unified** | **Jaccard** | **Chem** | **SSL** | **P@10** | **R@10** | **F1@10** | **DR_P@10** | **DR_R@10** | **DR_F1@10** |
| **All (HMC-GNN)** | ✓   | ✓   | ✓   | ✓   | ✓   | **0.1853** | **0.3435** | **0.1751** | \-  | \-  | \-  |
| W/O Top-K |     | ✓   | ✓   | ✓   | ✓   | 0.1258 | 0.1402 | 0.0973 | \-32.11% | \-59.18% | \-44.43% |
| W/O Unified | ✓   |     | ✓   | ✓   | ✓   | 0.1466 | 0.2028 | 0.1223 | \-20.89% | \-40.96% | \-30.15% |
| W/O Jaccard | ✓   | ✓   |     | ✓   | ✓   | 0.1788 | 0.3128 | 0.1659 | \-3.51% | \-8.94% | \-5.25% |
| W/O Chem | ✓   | ✓   | ✓   |     |     | 0.1670 | 0.2924 | 0.1547 | \-9.88% | \-14.88% | \-11.65% |
| W/O SSL | ✓   | ✓   | ✓   | ✓   |     | 0.1763 | 0.3158 | 0.1659 | \-4.86% | \-8.06% | \-5.25% |

Overall, the results demonstrate that removing any of the aforementioned components leads to a decline in model performance, underscoring their significance in HMC-GNN. Specifically, the following observations can be made:

**(1) Impact of Graph Topological Interventions (RQ2):**  
The ablation of structural pruning components results in substantial performance degradation. Notably, removing the Top-K filter (W/O Top-K) significantly reduces performance, yielding the most severe drop (-59.18% in R@10 and -44.43% in F1@10). This confirms our previous observation that without proper truncation, globally dense hub herbs easily dominate the graph message-passing, leading to over-smoothing. Similarly, decoupling the unified graph (W/O Unified) causes the second-worst degradation (-30.15% in F1@10), highlighting the necessity of processing heterogeneous entities within a unified topological space to preserve high-order multi-hop associations.

**(2) Impact of Multimodal Fusion and Semantic Alignment (RQ3):**  
The incorporation of multimodal features and their fusion strategies significantly affects recommendation accuracy. Completely omitting chemical formulations (W/O Chem) leads to a noticeable performance decline (-11.65% in F1@10), proving the value of integrating micro-level biochemical metrics. Furthermore, utilizing simple concatenation instead of the designed contrastive alignment (W/O SSL) fails to recover full performance. This indicates that direct concatenation struggles to bridge the extreme dimensional differences between discrete TCM properties and continuous biochemical vectors.

**(3) Mitigation of Hub-node Bias and Graph Construction Strategy (RQ2):**
To verify our model's robustness against popularity bias, we conducted an explicit long-tail distribution analysis. Based on global occurrence frequencies, the 395 herbs were stratified into three groups: Head (Top 20%, 79 ubiquitous herbs), Mid (20%-60%, 158 herbs), and Tail (Bottom 40%, 158 rare herbs). We evaluated the Recall@10 across these groups for HMC-GNN and the baselines. The results reveal a striking limitation in conventional bipartite models: **BSGAM** completely collapses on the Tail group (Recall@10 = 0.0000) and performs poorly on the Mid group (0.0095), while achieving high Head performance (0.4901). This indicates BSGAM acts as a generic "popularity recommender," heavily favoring ubiquitous herbs at the expense of specific therapeutic signals. KDHR maintains some Tail performance (0.1048), but falls significantly behind in the Mid group (0.1974). In contrast, **HMC-GNN** achieves a highly balanced distribution, drastically outperforming baselines in the Mid group (0.2449) and preserving strong Tail recommendation (0.1026). This proves our framework successfully learns fine-grained disease-herb semantics rather than merely exploiting statistical commonalities.

To understand exactly how the asymmetric Top-K Jaccard graph construction achieves this balance, we compared it against two extreme topological variants: **Dense Graph** (retaining all meta-path connections without Top-K pruning) and **No Edge** (removing all collaborative edges). The empirical data perfectly aligns with our theoretical hypothesis: the Dense Graph drives the Head Recall up to 0.5090 but degrades the Tail to 0.0724. This confirms that overly dense connections trigger severe topological over-smoothing, allowing structural hub nodes to contaminate and overshadow rare local therapeutic signals. Conversely, the No Edge formulation isolates nodes, stripping away the essential collaborative generalization ability, which also results in poor Tail performance (0.0744). By acting as a critical topological bottleneck, our **Top-K Jaccard** strategy successfully filters out spurious hub connections while preserving high-confidence localized synergistic clusters, achieving the most balanced performance across the entire spectrum.

### **Influence of Hyper-Parameters (RQ4)**

To gain deeper insights into the robustness and behavioral characteristics of our framework, we conduct a comprehensive parameter sensitivity analysis evaluating four critical hyperparameters: the cutoff size for asymmetric graph construction, the contrastive learning weight , the InfoNCE temperature coefficient , and the latent embedding dimension .

- Effect of Cutoff Size (Topological Information Bottleneck). In the heterogeneous collaborative graph, the hyperparameter dictates the strictness of the structural bottleneck. We fixed the co-occurrence out-degree on the disease side and varied the maximum out-degree of the herb-herb graph within . As illustrated in **Fig. 3(a)**, the performance curves (measured strictly by F1@5, F1@10, and F1@20) exhibit a distinct inverted-U shape, peaking precisely at . When is too small (e.g., ), the topology becomes overly sparse, leading to Information Fragmentation (isolated structural islands) that hinders the GNN from effectively propagating high-order synergistic signals. Conversely, an overly relaxed bottleneck () fails to filter out highly versatile hub herbs. Such indiscriminate dense connectivity triggers a severe Hairball Effect and message over-smoothing, rendering initially specific nodes indistinguishable in the latent space. Thus, strikes the optimal balance between preserving local semantic communities and suppressing generic graph noise.
- Effect of Contrastive Learning Weight (Multi-task Trade-off). The hyperparameters and govern the magnitude of the multimodal contrastive loss () relative to the dominant BPR recommendation loss. To investigate the system's response to the overall contrastive constraint intensity, we varied a base weight while maintaining their relative mathematical ratio (, ). As depicted in **Fig. 3(b)**, the model achieves absolute peak performance at before experiencing a noticeable decline. The underlying rationale is that the contrastive loss acts strictly as a prior regularizer. A moderate intensity () effectively guides the structure-only embeddings to softly align with deep physicochemical continuous manifolds, thereby preventing topological collapse. However, excessively large weights () cause rigorous spatial alignment to dominate gradient updates, overshadowing the primary bipartite recommendation objective. Fundamentally, this degenerates the model's focus from precise graph link prediction to mere unimodal proxy reconstruction, consequently leading to degraded recommendation metrics.
- Effect of Temperature Coefficient (Contrastive Hardness). The temperature scaling parameter in the InfoNCE loss controls the penalty strength assigned to hard negative samples during cross-modal alignment. We evaluated . As shown in **Fig. 3(c)**, the model yields optimal metrics at . An excessively high (e.g., ) causes the model to treat all negative samples equally, leading to an overly uniform feature distribution that blurs the distinct pharmacological boundaries (e.g., failing to geometrically separate Bitter and Pungent properties). Conversely, an extremely low forces the model to be overly sensitive to noise and local anomalous structures, resulting in training instability and poor overall generalization.
- Effect of Latent Embedding Dimension (Model Capacity). The embedding dimension determines the expressive capacity of the unified latent space. We tested . As presented in **Fig. 3(d)**, performance consistently improves as increases from 32 to 128. This demonstrates that higher dimensions are required to sufficiently accommodate both the sparse graph topological structure and the rich, pre-trained chemical semantics (projected from an initial 768-dimensional space). However, beyond , the performance gains plateau and begin to slightly degrade. We attribute this to model over-parameterization, which inevitably increases the risk of overfitting on the limited, high-sparsity disease-herb interaction pairs.

Fig. 3. Hyperparameter sensitivity analysis of the proposed HMC-GNN framework. **(a)** the asymmetric topological cutoff size ; **(b)** the overall contrastive learning weight ; **(c)** the InfoNCE temperature coefficient ; and **(d)** the latent embedding dimension

### Case Study and Representation Visualization (RQ5)

To address RQ5, we intuitively investigate whether the learned latent space genuinely aligns multimodal TCM principles and how it translates into verifiable clinical recommendations, moving beyond purely macroscopic numerical metrics.

**Latent Space Visualization**

A fundamental challenge in heterogeneous graphs is resolving the "Modality Gap" to ensure models capture authentic pharmacological logic rather than exploiting topological shortcuts. **Fig. 4** presents the 3D PCA projection of the latent representation vectors.

As observed in **Fig. 4(a)** (**W/O SSL**), a severe modality gap persists: chemically generated features (blue squares) remain rigidly isolated from the structural manifolds (red dots). Conversely, under our dual-view mutual information maximization framework ( and ), **Fig. 4(b)** exhibits a cohesive shared manifold where heterogeneous features are successfully fused.

Crucially, to verify whether this mathematical convergence holds pharmacological validity, we recolored the final herb embeddings based on opposing TCM attributes (Bitter vs. Pungent) in **Fig. 4(c)**. Distinct geometric boundaries spontaneously emerge without explicit flavor-classification supervision: cold-natured Bitter herbs cluster locally, opposing the diaphoretic Pungent subspace. This visual evidence confirms that our contrastive strategy successfully anchors topological embeddings to the deep isomorphic relationship between microscopic chemistry and macroscopic TCM theories.

Fig. 4. 3D PCA visualizations of the multimodal latent representations. (a) Representation distribution using direct concatenation (w/o SSL). (b) Representation distribution under the proposed contrastive alignment framework (w/ SSL). (c) Final herb embeddings recolored based on TCM flavor attributes (Bitter vs. Pungent).

**Clinical Case Study**

Given the inherent flexibility of TCM prescriptions, purely metric-driven evaluations may not fully reflect a model's clinical viability. Therefore, we present two real-world clinical cases (Type 2 Diabetes and Cough) in Table 5 to intuitively analyze the rationality of HMC-GNN's herb recommendations.

As shown in Table 6, the recommended herbs exhibit therapeutic effects that theoretically align with the given diseases. For Case 1 (Type 2 Diabetes), the model achieves a 100% precision rate. The recommended herbs primarily function to "generate fluid" and "tonify Qi", which perfectly target the core pathogenesis of diabetes. For instance, _Brucea Fruit_ is prescribed for clearing heat, while _Red Ginseng_ is used to tonify Primal Qi. Furthermore, although some recommendations in Case 2 (Cough) do not exactly match the historical ground truth, they still contribute to treating the related condition theoretically. For example, the model recommends _Pokeweed Root_ (商陆) as a substitute. While absent from the original prescription, this herb is recognized in TCM practice for addressing complex respiratory symptoms or related fluid accumulations. Thus, it serves as a reasonable theoretical alternative when constructing a comprehensive formulation for intractable coughs. Through this qualitative analysis, we conclude that HMC-GNN can provide highly accurate and pharmacologically rational herb recommendations based on disease inputs.

Table 5

HMC-GNN herb recommendation case.

|     |     |     |     |
| --- | --- | --- | --- |
| **Case1** |     |     |     |
| Disease (TCM Syndrome) | **Herb Set** |     |     |
| Ground Truth | HMC-GNN | Effects |
| Type 2 Diabetes<br><br>（消渴症） | Brucea Fruit (鸦胆子),<br><br>Achyranthes Root (牛膝),<br><br>Red Ginseng (红参),<br><br>Reishi Mushroom (灵芝),<br><br>Lycopus (泽兰) | **Brucea Fruit** (鸦胆子),<br><br>**Achyranthes Root** (牛膝),<br><br>**Red Ginseng** (红参),<br><br>**Reishi Mushroom** (灵芝),<br><br>**Lycopus** (泽兰) | Clear Heat and Generate Fluid.<br><br>Activate Blood and Tonify Liver/Kidney.<br><br>Greatly Tonify Primal Qi.<br><br>Replenish Qi and Nourish Strength<br><br>Activate Blood and Resolve Stasis. |
| Precision@5: 1.0000 |     |     |     |
| **Case2** |     |     |     |
| Disease (TCM Syndrome) | **Herb Set** |     |     |
| Ground Truth | HMC-GNN | Effects |
| Cough<br><br>（咳嗽） | Rehmannia (地黄),<br><br>Asparagus Root (天冬),<br><br>Cornus (山茱萸),<br><br>Chinese Yam (山药),<br><br>Jujube (大枣),<br><br>Atractylodes (白术),<br><br>Purslane (马齿苋),<br><br>Cocklebur (苍耳子),<br><br>Typhaceae (蒲黄)\*,<br><br>Codonopsis (党参) | **Rehmannia** (地黄),<br><br>**Asparagus Root** (天冬),<br><br>**Cornus** (山茱萸),<br><br>**Chinese Yam** (山药),<br><br>**Jujube** (大枣),<br><br>**Atractylodes** (白术),<br><br>**Purslane** (马齿苋),<br><br>**Cocklebur** (苍耳子),<br><br>Pokeweed Root (商陆)\*,<br><br>**Codonopsis** (党参) | Clear Heat and Generate Fluid.<br><br>Nourish Yin, Moisten Lungs and Generate Fluid.<br><br>Tonify Liver and Kidney.<br><br>Generate Fluid, Benefit Lungs and Tonify Spleen.<br><br>Tonify Spleen/Stomach and Moisten Lungs.<br><br>Tonify Spleen Qi and Resolve Phlegm/Dampness.<br><br>Clear Heat and Detoxify.<br><br>Dispel Wind-Cold and Open Nasal Passages.<br><br>Dispel Water Retention and Dissipate Nodules.<br><br>Tonify Spleen, Benefit Lungs and Replenish Middle Qi. |
| Precision@10: 0.9000 |     |     |     |

## Discussion and Conclusion

### Discussion

**Advantages and Clinical Prospects of HMC-GNN**

HMC-GNN shows strong empirical performance in disease-driven TCM prescription recommendation and provides a structured way to integrate heterogeneous therapeutic knowledge. The asymmetric Top-K graph construction helps alleviate hub-node bias and reduce over-smoothing during graph propagation, thereby preserving more discriminative local therapeutic signals. In addition, the adaptive multimodal fusion mechanism enables the joint modeling of structural information, macroscopic TCM attributes, and microscopic chemical descriptors in a shared latent space. Visualization results further suggest that the learned representations capture meaningful multimodal structure.

From an application perspective, these properties may improve the interpretability of the recommendation process compared with methods that rely on a single modality or simplified graph structure. Nevertheless, HMC-GNN should be regarded as an auxiliary reference framework rather than a standalone clinical decision system, and its practical use still requires further expert evaluation and clinical validation.

**Limitations and Future Work**

Several limitations should be noted. First, the current framework focuses on herb set recommendation and does not model dosage, which is important in real TCM prescription formulation. Second, the graph construction strategy relies on fixed Top-K pruning and predefined thresholds, which may remove potentially useful long-range relations. Third, disease representation is currently based mainly on textual and graph-level information, without incorporating richer patient-level observations such as tongue images, pulse signals, or temporal symptom evolution. Future work will explore dosage-aware recommendation, adaptive graph construction, and patient-level multimodal modeling to improve clinical relevance and generalization.

### Conclusion

This paper proposes HMC-GNN, a multimodal graph neural network for disease-driven TCM prescription recommendation. By combining asymmetric graph construction, multimodal feature fusion, and self-supervised alignment, the proposed framework addresses hub-node bias and improves cross-modal representation learning in sparse disease-input settings. Experiments on the ETCM-HERBKG benchmark show that HMC-GNN consistently outperforms several competitive baselines across major evaluation metrics. Moreover, representation visualization provides qualitative evidence that the model learns more structured and interpretable multimodal embeddings. Future work will further extend the framework toward dosage-aware and patient-level multimodal prescription recommendation.

## 参考文献

1.  Hu Q, Yu T, Li J, Yu Q, Zhu L, Gu Y. End-to-end syndrome differentiation of Yin deficiency and Yang deficiency in traditional Chinese medicine. Comput Methods Programs Biomed 2019;174:9–15.
2.  Li, FS., Weng, JK. Demystifying traditional herbal medicine with modern approach. Nature Plants 3, 17109 (2017). https://doi.org/10.1038/nplants.2017.109
3.  Hao P, Jiang F, Cheng J, et al. Traditional Chinese medicine for cardiovascular disease: evidence and potential mechanisms\[J\]. Journal of the American College of Cardiology, 2017, 69(24): 2952-2966.
4.  Hopkins A L. Network pharmacology: the next paradigm in drug discovery\[J\]. Nature chemical biology, 2008, 4(11): 682-690.
5.  Zhou X, Seto S W, Chang D, et al. Synergistic effects of Chinese herbal medicine: a comprehensive review of methodology and current research\[J\]. Frontiers in pharmacology, 2016, 7: 201.
6.  Zeng J, Jia X. Quantifying compatibility mechanisms in traditional Chinese medicine with interpretable graph neural networks\[J\]. Journal of Pharmaceutical Analysis, 2025: 101342.
7.  Yuan H, Ma Q, Ye L, et al. The traditional medicine and modern medicine from natural products\[J\]. Molecules, 2016, 21(5): 559.
8.  Yin L, Xue X, Pan S, et al. Artificial intelligence in Traditional Chinese Medicine: systematic insights from data mining, large language models, and multimodal fusion\[J\]. Acta Materia Medica, 2025, 4(4): 674-697.
9.  Ren Y, Luo X, Wang Y, et al. Large language models in traditional Chinese medicine: a scoping review\[J\]. Journal of Evidence‐Based Medicine, 2025, 18(1): e12658.
10. Pan D, Guo Y, Fan Y, et al. Development and application of traditional Chinese medicine using AI machine learning and deep learning strategies\[J\]. The American journal of Chinese medicine, 2024, 52(03): 605-623.
11. Jin Y, Zhang W, He X, et al. Syndrome-aware herb recommendation with multi-graph convolution network\[C\]//2020 IEEE 36th international conference on data engineering (ICDE). IEEE, 2020: 145-156.
12. Jia Q, Zhang D, Yang S, et al. Traditional Chinese medicine symptom normalization approach leveraging hierarchical semantic information and text matching with attention mechanism\[J\]. Journal of biomedical informatics, 2021, 116: 103718.
13. Zhai X, Wang X, Wang L, et al. Treating different diseases with the same method—a traditional Chinese medicine concept analyzed for its biological basis\[J\]. Frontiers in Pharmacology, 2020, 11: 946.
14. Yu S, Liang Z, Wu Q, et al. A novel diagnostic and therapeutic strategy for cancer patients by integrating Chinese medicine syndrome differentiation and precision medicine\[J\]. Chinese Journal of Integrative Medicine, 2022, 28(10): 867-871.
15. Lu Z H, Yang C L, Yang G G, et al. Efficacy of the combination of modern medicine and traditional Chinese medicine in pulmonary fibrosis arising as a sequelae in convalescent COVID-19 patients: a randomized multicenter trial\[J\]. Infectious diseases of poverty, 2021, 10(1): 31.
16. Zhang P, Zhang D, Zhou W, et al. Network pharmacology: towards the artificial intelligence-based precision traditional Chinese medicine\[J\]. Briefings in bioinformatics, 2024, 25(1): bbad518.
17. Li L, Yang L, Yang L, et al. Network pharmacology: a bright guiding light on the way to explore the personalized precise medication of traditional Chinese medicine\[J\]. Chinese medicine, 2023, 18(1): 146.
18. Wu T, Ren H, Li P, et al. Graph information bottleneck\[J\]. Advances in Neural Information Processing Systems, 2020, 33: 20437-20448.
19. Liu Z, Yang J, Chen K, et al. TCM-KDIF: an information interaction framework driven by knowledge–data and its clinical application in traditional Chinese medicine\[J\]. IEEE Internet of Things Journal, 2024, 11(11): 20002-20014.
20. Zhou W, Yang K, Zeng J, et al. FordNet: recommending traditional Chinese medicine formula via deep neural network integrating phenotype and molecule\[J\]. Pharmacological research, 2021, 173: 105752.
21. Zhou X, Chen S, Liu B, et al. Development of traditional Chinese medicine clinical data warehouse for medical knowledge discovery and decision support\[J\]. Artificial Intelligence in medicine, 2010, 48(2-3): 139-152.
22. Ma J, Wang Z, Guo H, et al. Mining Syndrome Differentiating Principles from Traditional Chinese Medicine Clinical Data\[J\]. Computer Systems Science & Engineering, 2022, 40(3).
23. Zhang S, Wang W, Pi X, et al. Advances in the application of traditional Chinese medicine using artificial intelligence: a review\[J\]. The American journal of Chinese medicine, 2023, 51(05): 1067-1083.
24. Li W, Yang Z. Exploration on generating traditional Chinese medicine prescriptions from symptoms with an end-to-end approach\[C\]//CCF International Conference on Natural Language Processing and Chinese Computing. Cham: Springer International Publishing, 2019: 486-498.
25. Yang K, Dong X, Zhang S, et al. PresRecRF: Herbal prescription recommendation via the representation fusion of large TCM semantics and molecular knowledge\[J\]. Phytomedicine, 2024, 135: 156116.
26. Wang D, Liu P, Zheng Y, et al. Heterogeneous graph neural networks for extractive document summarization\[C\]//Proceedings of the 58th annual meeting of the association for computational linguistics. 2020: 6209-6219.
27. Jin Y, Ji W, Zhang W, et al. A KG-enhanced multi-graph neural network for attentive herb recommendation\[J\]. IEEE/ACM transactions on computational biology and bioinformatics, 2021, 19(5): 2560-2571.
28. Yang Y, Rao Y, Yu M, et al. Multi-layer information fusion based on graph convolutional network for knowledge-driven herb recommendation\[J\]. Neural Networks, 2022, 146: 1-10.
29. Tang X, Tang Y, Liu X, et al. Utilizing semantically enhanced self-supervised graph convolution and multi-head attention fusion for herb recommendation\[J\]. Artificial Intelligence in Medicine, 2025, 164: 103112.
30. Wu Z, Guo K, Luo E, et al. Medical long-tailed learning for imbalanced data: Bibliometric analysis\[J\]. Computer Methods and Programs in Biomedicine, 2024, 247: 108106.
31. Li Q, Han Z, Wu X M. Deeper insights into graph convolutional networks for semi-supervised learning\[C\]//Proceedings of the AAAI conference on artificial intelligence. 2018, 32(1).
32. Chen D, Lin Y, Li W, et al. Measuring and relieving the over-smoothing problem for graph neural networks from the topological view\[C\]//Proceedings of the AAAI conference on artificial intelligence. 2020, 34(04): 3438-3445.
33. Zhou X, Dong X, Li C, et al. TCM-FTP: Fine-tuning large language models for herbal prescription prediction\[C\]//2024 IEEE international conference on bioinformatics and biomedicine (BIBM). IEEE, 2024: 4092-4097.
34. Mswahili M E, Jeong Y S. Transformer-based models for chemical SMILES representation: A comprehensive literature review\[J\]. Heliyon, 2024, 10(20).
35. Baltrušaitis T, Ahuja C, Morency L P. Multimodal machine learning: A survey and taxonomy\[J\]. IEEE transactions on pattern analysis and machine intelligence, 2018, 41(2): 423-443.
36. Xin W, Zi-Yi W, Shao L I. TCM network pharmacology: a new trend towards combining computational, experimental and clinical approaches\[J\]. Chinese journal of natural medicines, 2021, 19(1): 1-11.
37. Zhai Y, Liu L, Zhang F, et al. Network pharmacology: a crucial approach in traditional Chinese medicine research\[J\]. Chinese medicine, 2025, 20(1): 8.
38. Zhang Y, Li X, Shi Y, et al. ETCM v2. 0: an update with comprehensive resource and rich annotations for traditional Chinese medicine\[J\]. Acta Pharmaceutica Sinica B, 2023, 13(6): 2559-2571.
39. Chithrananda S, Grand G, Ramsundar B. ChemBERTa: large-scale self-supervised pretraining for molecular property prediction\[J\]. arXiv preprint arXiv:2010.09885, 2020.
40. Rogers D, Hahn M. Extended-connectivity fingerprints\[J\]. Journal of chemical information and modeling, 2010, 50(5): 742-754.
41. Yao, L., Zhang, Y., Wei, B., Zhang, W., Jin, Z., 2018. A topic modeling approach for traditional chinese medicine prescriptions. IEEE Trans. Knowl. Data Eng. 30 (6), 1007–1021.
42. Wu, Y., Zhang, F., Yang, K., Fang, S., Bu, D., Li, H., Sun, L., Hu, H., Gao, K., Wang, W., Zhou, X., Zhao, Y., Chen, J., 2019. SymMap: an integrative database of traditional chinese medicine enhanced by symptom mapping. Nucleic Acids Res. 47 (D1), D1110–D1117.
43. Wang X, Zhang Y, Wang X, et al. A knowledge graph enhanced topic modeling approach for herb recommendation\[C\]//International conference on database systems for advanced applications. Cham: Springer International Publishing, 2019: 709-724.
44. Wang Z, Poon J, Poon S. Tcm translator: A sequence generation approach for prescribing herbal medicines\[C\]//2019 IEEE international conference on bioinformatics and biomedicine (BIBM). IEEE, 2019: 2474-2480.
45. Liu Z, Zheng Z, Guo X, et al. Attentiveherb: a novel method for traditional medicine prescription generation\[J\]. IEEE Access, 2019, 7: 139069-139085.
46. Li C, Liu D, Yang K, et al. Herb-know: knowledge enhanced prescription generation for traditional Chinese medicine\[C\]//2020 IEEE international conference on bioinformatics and biomedicine (BIBM). IEEE, 2020: 1560-1567.
47. Liu, Z., Luo, C., Fu, D., Gui, J., Zheng, Z., Qi, L., Guo, H., 2022. A novel transfer learning model for traditional herbal medicine prescription generation from unstructured resources and knowledge. Artif. Intell. Med. 124, 102232.
48. Zhao, Z., Ren, X., Song, K., Qiang, Y., Zhao, J., Zhang, J., Han, P., 2023. Pregenerator: tcm prescription recommendation model based on retrieval and generation method. IEEE Access 11, 103679–103692.
49. Hou, J., Song, P., Zhao, Z., Qiang, Y., Zhao, J., Yang, Q., 2023. TCM Prescription Generation via Knowledge Source Guidance Network Combined with Herbal Candidate Mechanism. Computational and mathematical methods in medicine 2023, 3301605.
50. Jin, Y., Zhang, W., He, X., Wang, X., Wang, X., 2020. Syndrome-aware herb recommendation with multi-graph convolution network. 2020 IEEE 36th International Conference on Data Engineering (ICDE). IEEE, pp. 145–156
51. Zhao, W., Lu, W., Li, Z., Fan, H., Yang, Z., Lin, X., Li, C., 2022. Tcm herbal prescription recommendation model based on multi-graph convolutional network. J. Ethnopharmacol. 297, 115109.
52. Yan, J., Wen, Z., Zou, B., 2022. Heterogeneous graph construction and node representation learning method of treatise on febrile diseases based on graph convolutional network. Digit. Chinese Med. 5 (4), 419–428.
53. Yang, X., Ding, C., 2023. Smrgat: a traditional chinese herb recommendation model based on a multi-graph residual attention network and semantic knowledge fusion. J. Ethnopharmacol., 116693
54. Rendle S, Freudenthaler C, Gantner Z, et al. BPR: Bayesian personalized ranking from implicit feedback\[J\]. arXiv preprint arXiv:1205.2618, 2012.
55. You Y, Chen T, Sui Y, et al. Graph contrastive learning with augmentations\[J\]. Advances in neural information processing systems, 2020, 33: 5812-5823.
56. Abdollahpouri H, Burke R, Mobasher B. Controlling popularity bias in learning-to-rank recommendation\[C\]//Proceedings of the eleventh ACM conference on recommender systems. 2017: 42-46.
57. Su Y, Zhang R, M. Erfani S, et al. Neural graph matching based collaborative filtering\[C\]//Proceedings of the 44th international ACM SIGIR conference on research and development in information retrieval. 2021: 849-858.
58. Dong X, Zheng Y, Shu Z, Chang K, Xia J, Zhu Q, Zhong K, Wang X, Yang K, Zhou X. TCMPR: TCM prescription recommendation based on subnetwork term mapping and deep learning. BioMed Res Int 2022;2022(1):4845726.