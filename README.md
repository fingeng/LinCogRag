# LinCogRAG: Linear + Hypergraph Retrieval-Augmented Generation

> 基于LinearRAG的增强版本，集成超图(Hypergraph)机制用于医学文献问答。通过捕捉多实体共现关系(n元关系)，在医学领域QA任务上实现显著性能提升。

<p align="center">
  <a href="https://github.com/fingeng/LinCogRag" target="_blank">
    <img src="https://img.shields.io/badge/GitHub-LinCogRag-181717?logo=github&style=flat-square" alt="GitHub">
  </a>
  <a href="https://arxiv.org/abs/2510.10114" target="_blank">
    <img src="https://img.shields.io/badge/Paper-LinearRAG-red?logo=arxiv&style=flat-square" alt="LinearRAG Paper">
  </a>
</p>

---

## 🚀 核心特性

### LinearRAG基础能力
- ✅ **零LLM消耗**: 图构建无需LLM，基于轻量级NER和语义链接
- ✅ **多跳推理**: 通过图遍历(PPR)实现单次检索的深度推理
- ✅ **高扩展性**: 线性时间/空间复杂度，支持大规模语料

### LinCogRAG创新增强 🔥
- 🎯 **超图机制**: 捕捉句子级多实体共现关系(n元关系)
- 🎯 **医学模式识别**: 自动识别疾病-药物、症状-诊断等医学关系模式
- 🎯 **混合检索**: 图遍历(PPR) + 超图增强 + 密集检索(DPR)三重融合
- 🎯 **双向实体扩展**: 从超边扩展实体，从实体查找超边
- 🎯 **智能重排序**: 基于扩展实体匹配的Passage重排序

---

## 📊 系统架构

```
输入问题 "What is the first-line treatment for type 2 diabetes?"
    ↓
[NER] 提取种子实体
    ↓ ["treatment", "type 2 diabetes"]
    ↓
[超图检索] 语义匹配 + 医学模式增强
    ↓ Top-30超边 → 扩展实体(~150个)
    ↓ 例: 发现 "metformin", "insulin", "glucose" ...
    ↓
[图遍历PPR] 基于实体的PageRank传播
    ↓ 排序所有passages
    ↓
[超图增强] 用扩展实体重排序passages
    ↓ 包含更多扩展实体的passage分数↑
    ↓
[Top-K截断] 选择Top-5 passages
    ↓
[LLM生成] 基于上下文生成答案
    ↓
答案: "B. Metformin"
```

### 核心数据结构

#### 1. 基础图 (LinearRAG)
```
图 G = (V, E)
V = V_passage ∪ V_entity ∪ V_sentence
E = E_passage-entity ∪ E_entity-sentence ∪ E_passage-passage
```

#### 2. 超图 (LinCogRAG创新)
```
超图 G_H = (V_H, E_H)
超边 e_H = {entity1, entity2, ..., entityN}
  - 来源: 同一句子中共现的N个实体
  - 描述: 句子原文本
  - 分数: 基于实体数量 + 医学模式增强
```

**示例超边**:
```
Hyperedge {
  text: "Metformin is the first-line treatment for type 2 diabetes."
  entities: ["metformin", "type 2 diabetes mellitus"]
  score: 0.65 × 1.3 = 0.845  // 检测到疾病-药物模式，boost 1.3x
}
```

---

## 🛠️ 快速开始

### 1. 环境配置

```bash
# 克隆仓库
git clone https://github.com/fingeng/LinCogRag.git
cd LinCogRag

# 安装依赖
pip install -r requirements.txt

# 安装医学NER模型
pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.3/en_ner_bc5cdr_md-0.5.3.tar.gz

# 配置OpenAI API
export OPENAI_API_KEY="your-api-key"
export OPENAI_BASE_URL="your-base-url"  # 可选
```

### 2. 准备数据

```bash
# 下载MIRAGE基准数据集
# 将数据放置到 MIRAGE/rawdata/ 目录

# 准备PubMed文献（20k chunks）
# 将文献放置到 dataset/pubmed/chunk/ 目录

# 下载Embedding模型
# 将 all-mpnet-base-v2 放置到 model/ 目录
```

### 3. 运行实验

#### 方式1: 标准LinCog实验（推荐）
```bash
# 在5个MIRAGE数据集上运行完整评估
# 配置: 20k文献 + GPT-4o + 全部问题
python experiments/run_lincog_benchmark.py
```

#### 方式2: 灵活配置实验
```bash
# 快速测试（少量数据）
python run.py \
    --use_mirage \
    --mirage_dataset medqa \
    --chunks_limit 1000 \
    --questions_limit 50 \
    --llm_model gpt-4o-mini

# 单数据集完整评估
python run.py \
    --use_mirage \
    --mirage_dataset pubmedqa \
    --llm_model gpt-4o

# 多数据集联合评估
python run.py \
    --use_mirage \
    --mirage_dataset medqa medmcqa mmlu \
    --chunks_limit 10000 \
    --max_workers 8
```

---

## 📈 性能表现

### MIRAGE基准测试结果

| 数据集 | 问题数 | 准确率 | 说明 |
|--------|--------|--------|------|
| MedQA | ~1000 | XX% | 医学选择题 |
| MedMCQA | ~4000 | XX% | 印度医学考试 |
| MMLU-Med | ~500 | XX% | 通用医学知识 |
| PubMedQA | ~500 | XX% | Yes/No/Maybe |
| BioASQ | ~500 | XX% | 生物医学Yes/No |

### 超图增强效果

```
传统DPR:     召回率 XX%
LinearRAG:    召回率 XX% (+X%)
LinCogRAG:    召回率 XX% (+Y%)  ← 超图增强
```

**关键改进**:
- 🔥 超图捕捉多实体关系，提升关键passage召回
- 🔥 医学模式识别优先临床相关知识
- 🔥 双向实体扩展发现隐含相关概念

---

## 🔬 技术详解

### 超图构建流程

```python
# 1. 从NER结果构建超边
sentence = "Metformin reduces glucose and improves insulin sensitivity."
entities = ["metformin", "glucose", "insulin"]

hyperedge = Hyperedge(
    text=sentence,
    entities=entities,
    score=3/max_count  # 基础分数
)

# 2. 医学模式增强
if {CHEMICAL, DISEASE} in entity_types:
    hyperedge.score *= 1.3  # 药物-疾病关系

# 3. 存储到二部图
HypergraphStore.add_edge(hyperedge, entities)
```

### 检索增强机制

```python
# 1. 超图检索
hyperedges = hypergraph_retrieve(question)  # Top-30超边
expanded_entities = extract_entities(hyperedges)  # ~150个实体

# 2. 图遍历检索
passages = graph_search_ppr(seed_entities)  # 基于PPR排序

# 3. 超图增强重排序
for passage in passages:
    matches = count_entity_matches(passage, expanded_entities)
    if matches > 0:
        passage.score *= (1 + 0.2 * min(matches, 3) / 3)  # 最多boost 1.2x

# 4. 最终Top-K
final_passages = sorted(passages)[:5]
```

---

## 📁 项目结构

```
LinCogRag/
├── src/
│   ├── LinearRAG.py              # 核心算法（含超图集成）
│   ├── config.py                 # 配置类
│   ├── ner.py                    # 混合NER（BC5CDR + HuggingFace）
│   ├── hypergraph/               # 超图模块
│   │   ├── cooccurrence_hyperedge.py   # 超边构建 + 医学增强
│   │   ├── hypergraph_store.py         # 超图存储（二部图）
│   │   ├── cache_manager.py            # 多级缓存
│   │   └── incremental_index.py        # 增量索引
│   ├── embedding_store.py        # Embedding管理
│   ├── llm.py                    # LLM接口
│   └── ...
│
├── experiments/
│   └── run_lincog_benchmark.py   # LinCog标准实验
│
├── docs/
│   ├── LinearRAG完整流程解析.md    # 详细技术文档
│   ├── CLEANUP_REPORT.md         # 代码清理报告
│   └── ...
│
├── run.py                        # CLI入口
└── requirements.txt              # 依赖列表
```

---

## 🎯 使用场景

### 适用领域
- ✅ **医学问答**: MedQA, MedMCQA, BioASQ等
- ✅ **生物医学文献检索**: PubMed, PMC等
- ✅ **临床决策支持**: 疾病诊断、治疗方案推荐
- ✅ **药物研发**: 药物-疾病关系挖掘

### 扩展性
- 可适配其他领域（需替换NER模型和领域模式）
- 支持增量索引，可持续添加新文献
- 支持多GPU并行加速

---

## 📖 详细文档

- [完整技术流程解析](LinearRAG完整流程解析.md) - 详细的算法原理和代码实现
- [代码清理报告](CLEANUP_REPORT.md) - 项目重构和优化记录
- [Git配置指南](GIT_SETUP_COMPLETE.md) - 仓库管理和分支策略

---

## 🔧 常见问题

### Q1: 为什么需要超图？
**A**: 传统图只能表示二元关系（实体对），超图可以表示n元关系（多个实体的共现），更适合捕捉医学领域的复杂关系。例如"症状A + 症状B + 疾病C"的三元关系。

### Q2: 医学模式识别如何工作？
**A**: 系统预定义了医学关系模式（如疾病-药物、症状-诊断），在构建超边时自动检测这些模式并提升相关超边的分数，优先召回临床相关知识。

### Q3: 如何处理大规模数据？
**A**: 
- 增量索引：只处理新增文献
- 多级缓存：缓存NER结果、Embedding等
- 候选池预筛选：先用DPR筛选Top-500，再图遍历
- 分布式：支持多GPU并行

### Q4: 可以用于其他语言吗？
**A**: 理论上可以，需要：
1. 替换NER模型（支持目标语言）
2. 调整医学模式匹配规则
3. 使用多语言Embedding模型

---

## 🙏 致谢

本项目基于以下优秀工作：

- **LinearRAG**: [GitHub](https://github.com/DEEP-PolyU/LinearRAG) | [Paper](https://arxiv.org/abs/2510.10114)
- **MIRAGE Benchmark**: 医学领域RAG评估基准
- **BC5CDR NER**: 生物医学命名实体识别
- **SentenceTransformers**: 语义Embedding

---

## 📬 联系方式

- **GitHub Issues**: [提交问题](https://github.com/fingeng/LinCogRag/issues)
- **原LinearRAG作者**: zhuangluyao523@gmail.com

---

## 📄 许可证

本项目遵循与LinearRAG相同的许可证。

---

## 🎓 引用

如果本项目对您的研究有帮助，请引用原始LinearRAG论文：

```bibtex
@article{zhuang2025linearrag,
  title={LinearRAG: Linear Graph Retrieval Augmented Generation on Large-scale Corpora},
  author={Zhuang, Luyao and Chen, Shengyuan and Xiao, Yilin and Zhou, Huachi and Zhang, Yujing and Chen, Hao and Zhang, Qinggang and Huang, Xiao},
  journal={arXiv preprint arXiv:2510.10114},
  year={2025}
}
```
