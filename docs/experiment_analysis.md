# LinearRAG 实验流程与理论分析

## 一、实验配置

### 基本参数
- **数据集**: PubMed + MIRAGE + MMLU (医学问答)
- **问题数量**: 1089 个多选题
- **知识库规模**: 1 chunk (极小规模测试)
- **LLM 模型**: GPT-4o-mini
- **Embedding 模型**: all-mpnet-base-v2
- **NER 模型**: en_core_sci_scibert

---

## 二、系统运行流程

### 阶段 1: 索引构建 (Indexing Phase)

Input: PubMed 文献摘要
↓
[Step 1] Passage Embedding • 将文本转为 768 维向量 • 存储: passage_embedding.parquet
↓
[Step 2] NER 处理 • 使用 spaCy 提取实体 • 结果: 0 个实体 ⚠️ (关键问题)
↓
[Step 3] 图构建 • 节点: passages + entities • 边: 实体-段落关系
↓
Output: LinearRAG.graphml

#### 理论基础: 知识图谱增强检索
- **核心思想**: 将文本检索与知识图谱结合
- **优势**: 利用实体间关系进行多跳推理
- **现状**: 由于实体提取失败，退化为纯向量检索

---

### 阶段 2: 问答流程 (QA Phase)
For each question:
↓
[Step 1] 问题编码
question → embedding (768-dim vector)
↓
[Step 2] 种子实体提取
question → NER → entities
↓ (失败: 无实体)
↓
[Step 3] 检索策略选择
if entities exist:
→ Graph Search (实体图检索)
else:
→ Dense Retrieval (向量检索) ✓ (实际执行)
↓
[Step 4] 文档排序
计算 cosine_similarity(question_emb, passage_emb)
sorted_passages = top_k passages
↓
[Step 5] LLM 推理
context = sorted_passages
prompt = "Question: {q}\nContext: {c}\nAnswer:"
llm_output → 提取答案字母
↓
Output: prediction.json

---

## 三、核心理论

### 3.1 Dense Retrieval (稠密检索)

**公式**:
similarity(q, p) = (E_q · E_p) / (||E_q|| × ||E_p||) 
其中: • E_q: 问题的 embedding • E_p: 文档的 embedding • · : 向量点积 • ||·||: 向量模长

**优势**:
- 语义相似度计算
- 处理同义词和释义

**劣势**:
- 无法利用结构化知识
- 缺乏多跳推理能力

---

### 3.2 Graph-based Retrieval (图检索)

**理论框架**:
Seed Entity Matching
question_entities → KB_entities (通过 embedding 相似度)  2.  Entity Propagation (实体传播)
for iteration in [1, max_iterations]: ◦ 通过句子连接扩展实体 ◦ 衰减因子: score_new = score_old × similarity / tier   3.  Personalized PageRank (个性化 PageRank)
PPR(v) = α × reset(v) + (1-α) × Σ PPR(u) × w(u,v) 其中: ◦ reset(v): 初始概率分布 (基于实体权重) ◦ w(u,v): 边权重 ◦ α: 阻尼系数 (0.15)   4.  Passage Scoring
score(p) = λ × DPR_score(p) + log(1 + entity_bonus(p))