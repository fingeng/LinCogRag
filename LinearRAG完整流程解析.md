# LinearRAG + Hypergraph 完整流程解析

## 📋 目录
1. [系统架构概览](#系统架构概览)
2. [核心数据结构](#核心数据结构)
3. [索引构建流程](#索引构建流程)
4. [检索问答流程](#检索问答流程)
5. [具体示例详解](#具体示例详解)

---

## 系统架构概览

LinearRAG是一个基于**图结构**和**超图(Hypergraph)**的混合检索系统,用于医学文献问答。

### 核心思想
```
传统RAG: Query → Dense Retrieval → Top-K文档 → LLM生成
LinearRAG: Query → 实体识别 → 图遍历(PPR) → 超图增强 → Top-K文档 → LLM生成
```

### 关键创新点
1. **实体中心的图结构**: 将文档、实体、句子建模为图节点
2. **超图增强**: 捕捉多实体共现关系(n元关系)
3. **混合检索**: 结合密集检索(DPR)和图遍历(PPR)
4. **医学领域优化**: 专用NER、医学关系模式识别

---

## 核心数据结构

### 1. 基础图 (LinearRAG Graph)
```
图G = (V, E)
V = V_passage ∪ V_entity ∪ V_sentence
  - V_passage: 文档节点(passage chunks)
  - V_entity: 实体节点(医学实体)
  - V_sentence: 句子节点(包含实体的句子)

E = E_passage-entity ∪ E_entity-sentence ∪ E_passage-passage
  - E_passage-entity: 文档包含实体
  - E_entity-sentence: 句子包含实体
  - E_passage-passage: 相邻文档(顺序关系)
```

**节点属性**:
- `name`: hash_id (唯一标识)
- `content`: 文本内容
- `type`: 节点类型 (passage/entity/sentence)

**边权重**:
- passage→entity: TF(实体在文档中的频率归一化)
- entity→sentence: 共现关系
- passage→passage: 1.0(相邻文档)

### 2. 超图 (Hypergraph)
```
超图 G_H = (V_H, E_H)
V_H = 实体集合
E_H = 超边集合(hyperedges)

超边 e_H = {v1, v2, ..., vn} ⊆ V_H
  - 来源: 同一句子中共现的n个实体
  - 描述: 该句子的原文本
  - 置信度分数: score ∈ [0, 1.5]
```

**超边数据结构**:
```python
@dataclass
class Hyperedge:
    text: str                    # 原句子文本 (超边的自然语言描述)
    entities: List[str]          # 参与的实体列表 (≥2个)
    score: float                 # 置信度 (基于实体数量和医学模式)
    hash_id: str                 # 唯一标识
    entity_types: Dict[str,str]  # 实体类型映射
```

**存储结构(二部图表示)**:
```
超图存储为二部图 G_B = (V_B, E_B)
V_B = V_entity ∪ V_hyperedge
E_B = {(v, e_H) | v ∈ e_H}

这样可以高效查询:
- 给定实体 → 获取包含它的所有超边
- 给定超边 → 获取其包含的所有实体
```

### 3. 映射关系表
```python
# Passage ↔ Hyperedge
passage_to_hyperedge_ids: Dict[str, List[str]]
  # passage_hash_id → [hyperedge_hash_ids]

# Entity ↔ Hyperedge (二部图邻接表)
entity_to_hyperedges: Dict[str, Set[str]]
hyperedge_to_entities: Dict[str, Set[str]]

# Entity ↔ Sentence
entity_hash_id_to_sentence_hash_ids: Dict[str, List[str]]
sentence_hash_id_to_entity_hash_ids: Dict[str, List[str]]

# Hash ID ↔ Text
passage_embedding_store.hash_id_to_text: Dict[str, str]
entity_embedding_store.hash_id_to_text: Dict[str, str]
hyperedge_hash_to_text: Dict[str, str]
```

---

## 索引构建流程

### 完整流程图
```
输入: passages (20000篇PubMed文献)
  ↓
[Step 1] 插入passages → EmbeddingStore
  - 生成embeddings (SentenceTransformer)
  - 计算hash_id: md5(text)[:16]
  ↓
[Step 2-3] 加载已有NER结果 (增量索引)
  - 读取 ner_results.json
  - 识别新文档: new_hash_ids = current - existing
  ↓
[Step 4] 批量NER处理 (混合策略)
  - BC5CDR NER (spaCy): CHEMICAL, DISEASE
  - HuggingFace NER: 更多医学实体类型
  - 提取: passage_hash_id_to_entities, sentence_to_entities
  ↓
[Step 5] 保存NER结果 → ner_results.json
  ↓
[Step 6] 构建基础图节点和边
  - 提取: entity_nodes, sentence_nodes
  - 构建映射: entity_to_sentence, sentence_to_entity
  ↓
[Step 6.5] 🔥 构建超图 (HyperLinearRAG核心)
  │
  ├─ 6.5.1 从句子共现构建超边
  │   for sentence, entities in sentence_to_entities.items():
  │       if len(entities) >= 2:  # 至少2个实体
  │           hyperedge = Hyperedge(
  │               text=sentence,
  │               entities=list(entities),
  │               score=len(entities) / max_entity_count
  │           )
  │
  ├─ 6.5.2 医学模式增强分数
  │   for hyperedge in hyperedges:
  │       entity_types = get_entity_types(hyperedge.entities)
  │       # 检测医学关系模式
  │       if {DISEASE, CHEMICAL} ⊆ entity_types:
  │           score *= 1.3  # 疾病-药物关系
  │       if {SYMPTOM, DISEASE} ⊆ entity_types:
  │           score *= 1.2  # 症状-疾病关系
  │       # ... 更多模式
  │
  ├─ 6.5.3 存储到HypergraphStore
  │   - 构建二部图: entity ↔ hyperedge
  │   - 保存: hyperedges.pkl, metadata.json
  │
  └─ 6.5.4 构建passage-hyperedge映射
      for passage_hash_id, passage_text in passages:
          for hyperedge in hyperedges:
              if hyperedge.text in passage_text:
                  passage_to_hyperedge_ids[passage_hash_id].append(hyperedge.hash_id)
  ↓
[Step 7] 构建embeddings
  - entity_embeddings: (n_entities, 768)
  - sentence_embeddings: (n_sentences, 768)
  - passage_embeddings: (n_passages, 768)
  ↓
[Step 8] 构建igraph图
  - 添加所有节点: passages, entities, sentences
  - 添加边和权重
  - 保存: LinearRAG.graphml
  ↓
[Step 9] 🔥 加载超边embeddings
  - hyperedge_embeddings: (n_hyperedges, 768)
  - 用于后续检索时的语义匹配
  ↓
输出: 索引完成
  - 图结构: graph (igraph.Graph)
  - 超图: hypergraph_store
  - Embeddings: passage/entity/sentence/hyperedge
```

### 关键代码解析

#### Step 6.5: 超图构建
```python
def _build_hypergraph(self, sentence_to_entities, hash_id_to_passage):
    """从句子-实体共现构建超图"""
    
    # 1. 从共现构建超边
    hyperedges = self.hyperedge_builder.build_from_ner_results(
        sentence_to_entities  # {"句子": {"实体1", "实体2", ...}}
    )
    # 结果: [Hyperedge(text="句子", entities=["实体1",...], score=0.8), ...]
    
    # 2. 医学模式增强
    hyperedges = self.hyperedge_enhancer.enhance_hyperedges(hyperedges)
    # 检测医学关系模式,提升相关超边的score
    
    # 3. 存储到HypergraphStore (二部图)
    self.hypergraph_store.add_hyperedges(hyperedges)
    
    # 4. 构建passage→hyperedge映射
    passage_to_hyperedge_ids = {}
    for passage_hash_id, passage_text in hash_id_to_passage.items():
        for he in hyperedges:
            if he.text in passage_text:  # 句子包含在passage中
                passage_to_hyperedge_ids[passage_hash_id].append(he.hash_id)
    
    # 5. 保存
    self.hypergraph_store.save()
    self.passage_to_hyperedge_ids = passage_to_hyperedge_ids
```

---

## 检索问答流程

### 完整流程图
```
输入: question (例: "What is the first-line treatment for type 2 diabetes?")
  ↓
[Phase 1] Query处理
  ├─ 1.1 NER提取问题中的实体
  │   question_entities = spacy_ner.question_ner(question)
  │   # 例: ["type 2 diabetes", "treatment"]
  │
  ├─ 1.2 实体匹配 (语义相似度)
  │   question_entity_embeddings = encode(question_entities)
  │   similarities = dot(entity_embeddings, question_entity_embeddings)
  │   seed_entities = top_match_per_question_entity(similarities)
  │   # 例: [("diabetes mellitus type 2", 0.95), ("drug therapy", 0.88)]
  │
  └─ 1.3 Question embedding
      question_embedding = encode(question)
  ↓
[Phase 2] 🔥 混合检索 (HyperLinearRAG)
  │
  ├─ 2.1 超图检索 (hypergraph_retrieve)
  │   │
  │   ├─ 2.1.1 超边语义匹配
  │   │   hyperedge_scores = dot(hyperedge_embeddings, question_embedding)
  │   │   # 每个超边的embedding是其句子的embedding
  │   │
  │   ├─ 2.1.2 应用超边置信度
  │   │   for he_id, score in enumerate(hyperedge_scores):
  │   │       conf_score = hypergraph_store.get_hyperedge_score(he_id)
  │   │       hyperedge_scores[he_id] *= conf_score
  │   │
  │   ├─ 2.1.3 Top-K超边筛选
  │   │   top_hyperedges = argsort(hyperedge_scores)[:30]
  │   │   top_hyperedges = [he for he in top_hyperedges if score > 0.3]
  │   │
  │   └─ 2.1.4 双向实体扩展
  │       expanded_entities = set()
  │       # 方向1: 从超边扩展
  │       for he_id in top_hyperedges:
  │           entities = hypergraph_store.get_entities_by_hyperedge(he_id)
  │           expanded_entities.update(entities)
  │       
  │       # 方向2: 从种子实体扩展
  │       for entity_id in seed_entity_hash_ids:
  │           related_hyperedges = hypergraph_store.get_hyperedges_by_entity(entity_id)
  │           for he_id in related_hyperedges:
  │               entities = hypergraph_store.get_entities_by_hyperedge(he_id)
  │               expanded_entities.update(entities)
  │       
  │       # 结果: expanded_entities (扩展的实体集合,用于后续增强)
  │
  ├─ 2.2 LinearRAG图检索 (graph_search_with_seed_entities)
  │   │
  │   ├─ 2.2.1 候选池预筛选 (DPR)
  │   │   dpr_scores = dot(passage_embeddings, question_embedding)
  │   │   candidate_passages = argsort(dpr_scores)[:500]
  │   │
  │   ├─ 2.2.2 实体扩展 (图遍历)
  │   │   activated_entities = {seed_entities}
  │   │   for iteration in range(max_iterations):
  │   │       for entity in activated_entities:
  │   │           # 获取包含该实体的句子
  │   │           sentences = entity_hash_id_to_sentence_hash_ids[entity]
  │   │           
  │   │           # 计算句子与问题的相似度
  │   │           sentence_embeddings = get_embeddings(sentences)
  │   │           similarities = dot(sentence_embeddings, question_embedding)
  │   │           
  │   │           # Top-K相似句子
  │   │           top_sentences = argsort(similarities)[:5]
  │   │           
  │   │           # 从句子中提取新实体
  │   │           for sent in top_sentences:
  │   │               new_entities = sentence_hash_id_to_entity_hash_ids[sent]
  │   │               activated_entities.update(new_entities)
  │   │               entity_weights[new_entities] += scores
  │   │
  │   ├─ 2.2.3 Passage权重计算
  │   │   passage_weights = zeros(n_nodes)
  │   │   dpr_scores_normalized = min_max_normalize(dpr_scores)
  │   │   
  │   │   for passage_idx in candidate_passages:
  │   │       passage_hash_id = passage_hash_ids[passage_idx]
  │   │       passage_text = hash_id_to_text[passage_hash_id]
  │   │       
  │   │       # 实体匹配加成
  │   │       entity_bonus = 0
  │   │       for entity_id, (_, entity_score, tier) in activated_entities.items():
  │   │           entity_text = hash_id_to_text[entity_id]
  │   │           count = passage_text.count(entity_text)
  │   │           if count > 0:
  │   │               entity_bonus += entity_score * log(1 + count) / tier
  │   │       
  │   │       # 组合分数
  │   │       passage_weights[passage_idx] = (
  │   │           passage_ratio * dpr_scores_normalized[passage_idx] + 
  │   │           log(1 + entity_bonus)
  │   │       )
  │   │
  │   └─ 2.2.4 Personalized PageRank (PPR)
  │       node_weights = entity_weights + passage_weights
  │       pagerank_scores = graph.personalized_pagerank(
  │           reset=node_weights,  # 重启分布
  │           damping=0.85,
  │           weights='weight'
  │       )
  │       # 从图中传播重要性,平衡局部(实体)和全局(结构)信息
  │
  └─ 2.3 🔥 超图增强 (boost_passages_with_entities)
      # === 输入 ===
      # 1. 候选passages (来自PPR，已排序但未截断)
      # 2. expanded_entities (来自超图，~150个实体)
      
      boosted_scores = []
      for passage_hash_id, base_score in zip(passage_hash_ids, passage_scores):
          passage_text = hash_id_to_text[passage_hash_id]
          
          # 检查扩展实体匹配数
          entity_matches = 0
          for entity_id in expanded_entities:  # 来自超图
              entity_text = hypergraph_store.get_entity_text(entity_id)
              if entity_text.lower() in passage_text.lower():
                  entity_matches += 1
          
          # 应用增强系数 (包含越多扩展实体，boost越高)
          if entity_matches > 0:
              boost = 1 + (1.2 - 1) * min(entity_matches, 3) / 3
              base_score *= boost
          
          boosted_scores.append(base_score)
      
      # 重新排序所有passages
      sorted_passages = argsort(boosted_scores)[::-1]
      
      # === 输出 ===
      # 重排序后的passages (全部，尚未截断)
  ↓
[Phase 2.5] 🎯 最终Top-K筛选
  # 从重排序后的passages中取Top-K
  final_passage_hash_ids = sorted_passage_hash_ids[:5]  # retrieval_top_k=5
  final_passages = [hash_id_to_text[pid] for pid in final_passage_hash_ids]
  
  # === 关键点 ===
  # 扩展实体不是直接给LLM，而是用于重排序！
  # 只有重排序后Top-K的passages会进入prompt
  ↓
[Phase 3] LLM生成答案
  ├─ 3.1 构建超边上下文 (可选)
  │   if top_hyperedges:
  │       context = "[Medical Knowledge Facts]\n"
  │       for i, he_text in enumerate(top_hyperedges[:5]):
  │           context += f"{i+1}. {he_text}\n"
  │       # 将超边文本(关键句子)前置到上下文
  │
  ├─ 3.2 构建Prompt
  │   prompt = f"""Context:
  │   {hyperedge_context}
  │   
  │   {passage_1}
  │   
  │   {passage_2}
  │   ...
  │   
  │   Question: {question}
  │   YOUR RESPONSE MUST BE EXACTLY ONE LETTER: A, B, C, or D"""
  │
  └─ 3.3 LLM推理
      response = llm.infer(prompt)
      answer = parse_answer(response)  # 提取A/B/C/D
  ↓
输出: answer
```

### 关键算法详解

#### 1. 超图检索 (Hypergraph Retrieval)
```python
def hypergraph_retrieve(self, question_embedding, seed_entity_hash_ids):
    """超图检索返回: 相关超边、扩展实体"""
    
    # 1. 语义匹配: 问题 vs 超边
    hyperedge_scores = np.dot(
        self.hyperedge_embeddings,      # (n_hyperedges, 768)
        question_embedding.reshape(-1,1) # (768, 1)
    ).flatten()  # (n_hyperedges,)
    
    # 2. 应用超边置信度权重
    for idx, he_id in enumerate(self.hyperedge_hash_ids):
        conf_score = self.hypergraph_store.get_hyperedge_score(he_id)
        # conf_score来自医学模式增强 (1.0-1.5)
        hyperedge_scores[idx] *= conf_score
    
    # 3. Top-K筛选
    sorted_indices = np.argsort(hyperedge_scores)[::-1]
    top_hyperedges = []
    for idx in sorted_indices[:30]:  # top_k=30
        if hyperedge_scores[idx] < 0.3:  # threshold
            break
        he_id = self.hyperedge_hash_ids[idx]
        he_text = self.hypergraph_store.get_hyperedge_text(he_id)
        top_hyperedges.append((he_id, he_text, hyperedge_scores[idx]))
    
    # 4. 双向实体扩展
    expanded_entities = set()
    
    # 方向A: 从检索到的超边扩展
    for he_id, _, _ in top_hyperedges:
        entities = self.hypergraph_store.get_entities_by_hyperedge(he_id)
        expanded_entities.update(entities)
    
    # 方向B: 从种子实体扩展 (如果有)
    if seed_entity_hash_ids:
        for entity_id in seed_entity_hash_ids:
            # 获取包含该实体的所有超边
            related_hyperedges = self.hypergraph_store.get_hyperedges_by_entity(entity_id)
            for he_id in related_hyperedges:
                entities = self.hypergraph_store.get_entities_by_hyperedge(he_id)
                expanded_entities.update(entities)
    
    return (
        [text for _, text, _ in top_hyperedges],  # 超边文本
        [score for _, _, score in top_hyperedges], # 超边分数
        expanded_entities  # 扩展实体集合
    )
```

**为什么超图有效?**
- **捕捉n元关系**: 传统二元边只能表示实体对,超边可以表示"症状A + 症状B + 疾病C"的三元关系
- **句子级语境**: 超边保留原句子文本,提供完整语义上下文
- **医学知识增强**: 通过模式识别(如"疾病-药物"),提升临床相关超边的重要性
- **双向扩展**: 既可以从问题找超边,也可以从实体找超边,增加召回

#### 2. 混合检索融合
```python
def hybrid_retrieve(self, question, question_embedding, seed_entity_data):
    """LinearRAG图检索 + 超图增强"""
    seed_entity_indices, seed_entities, seed_entity_hash_ids, seed_entity_scores = seed_entity_data
    
    # Part 1: LinearRAG图检索 (PPR)
    if len(seed_entities) > 0:
        passage_hash_ids, passage_scores = self.graph_search_with_seed_entities(
            question_embedding, seed_entity_indices, seed_entities,
            seed_entity_hash_ids, seed_entity_scores
        )
    else:
        # 无实体时回退到密集检索
        sorted_indices, sorted_scores = self.dense_passage_retrieval(question_embedding)
        passage_hash_ids = [self.passage_embedding_store.hash_ids[idx] for idx in sorted_indices[:10]]
        passage_scores = sorted_scores[:10]
    
    # Part 2: 超图检索
    hyperedge_context = ""
    if self.use_hypergraph and self.hyperedge_embeddings is not None:
        hyperedge_texts, hyperedge_scores, expanded_entities = self.hypergraph_retrieve(
            question_embedding, seed_entity_hash_ids
        )
        
        # Part 3: 🔥 用扩展实体增强passage排序
        if expanded_entities:
            passage_hash_ids, passage_scores = self._boost_passages_with_entities(
                passage_hash_ids, passage_scores, expanded_entities
            )
        
        # Part 4: 格式化超边上下文 (用于LLM生成)
        if hyperedge_texts:
            hyperedge_context = self._format_hyperedge_context(
                hyperedge_texts, hyperedge_scores
            )
    
    return passage_hash_ids, passage_scores, hyperedge_context
```

---

## 具体示例详解

### 示例场景
- **Chunk**: "Metformin is the first-line pharmacological treatment for type 2 diabetes mellitus. It reduces hepatic glucose production and improves insulin sensitivity."
- **Query**: "What is the first-line treatment for type 2 diabetes?"

### Step-by-Step工作流

#### 1. 索引阶段

##### 1.1 Chunk插入
```
passage_text = "Metformin is the first-line pharmacological treatment..."
passage_hash_id = md5(passage_text)[:16]  # 例: "a3f5b2c8d1e9..."
passage_embedding = SentenceTransformer.encode(passage_text)  # (768,)
```

##### 1.2 NER提取
```python
# BC5CDR NER
entities = ["metformin", "type 2 diabetes mellitus", "glucose", "insulin"]

# HuggingFace NER (补充)
additional_entities = ["hepatic glucose production", "insulin sensitivity"]

# 合并
all_entities = ["metformin", "type 2 diabetes mellitus", "glucose", 
                "insulin", "hepatic glucose production", "insulin sensitivity"]

# 句子级提取
sentence_1 = "Metformin is the first-line pharmacological treatment for type 2 diabetes mellitus."
sentence_1_entities = {"metformin", "type 2 diabetes mellitus"}

sentence_2 = "It reduces hepatic glucose production and improves insulin sensitivity."
sentence_2_entities = {"glucose", "insulin", "hepatic glucose production", "insulin sensitivity"}
```

存储到:
```python
passage_hash_id_to_entities[passage_hash_id] = {
    "metformin", "type 2 diabetes mellitus", "glucose", "insulin", ...
}

sentence_to_entities = {
    "Metformin is the first-line...": {"metformin", "type 2 diabetes mellitus"},
    "It reduces hepatic glucose...": {"glucose", "insulin", ...}
}
```

##### 1.3 基础图构建
```
节点:
  - passage_node: (passage_hash_id, content="Metformin is the first-line...")
  - entity_nodes: (entity_hash_ids, content=["metformin", "type 2 diabetes mellitus", ...])
  - sentence_nodes: (sentence_hash_ids, content=[句子1, 句子2])

边:
  - passage → metformin: weight = 1/6 (出现1次,共6个实体)
  - passage → type 2 diabetes mellitus: weight = 1/6
  - entity → sentence: 共现关系
```

##### 1.4 超图构建 🔥
```python
# 句子1: 包含2个实体 → 构建超边
hyperedge_1 = Hyperedge(
    text="Metformin is the first-line pharmacological treatment for type 2 diabetes mellitus.",
    entities=["metformin", "type 2 diabetes mellitus"],
    score=2/4,  # 2个实体 / max_entity_count=4
    hash_id="he_abc123"
)

# 医学模式增强
entity_types = {"metformin": "CHEMICAL", "type 2 diabetes mellitus": "DISEASE"}
# 检测到模式: {CHEMICAL, DISEASE} → 疾病-药物关系
hyperedge_1.score *= 1.3  # 增强系数
# 最终score: (2/4) * 1.3 = 0.65

# 句子2: 包含4个实体 → 构建超边
hyperedge_2 = Hyperedge(
    text="It reduces hepatic glucose production and improves insulin sensitivity.",
    entities=["glucose", "insulin", "hepatic glucose production", "insulin sensitivity"],
    score=4/4 * 1.0,  # 无特殊医学模式
    hash_id="he_def456"
)

# 存储到HypergraphStore (二部图)
# Entity nodes: metformin, type 2 diabetes mellitus, glucose, insulin, ...
# Hyperedge nodes: he_abc123, he_def456
# Edges:
#   - (metformin, he_abc123)
#   - (type 2 diabetes mellitus, he_abc123)
#   - (glucose, he_def456)
#   - (insulin, he_def456)
#   - ...
```

映射关系:
```python
passage_to_hyperedge_ids[passage_hash_id] = ["he_abc123", "he_def456"]

entity_to_hyperedges = {
    "metformin": {"he_abc123"},
    "type 2 diabetes mellitus": {"he_abc123"},
    "glucose": {"he_def456"},
    "insulin": {"he_def456"},
    ...
}

hyperedge_to_entities = {
    "he_abc123": {"metformin", "type 2 diabetes mellitus"},
    "he_def456": {"glucose", "insulin", ...}
}
```

##### 1.5 超边Embedding
```python
hyperedge_embeddings = SentenceTransformer.encode([
    "Metformin is the first-line pharmacological treatment for type 2 diabetes mellitus.",
    "It reduces hepatic glucose production and improves insulin sensitivity."
])
# Shape: (2, 768)
```

---

#### 2. 检索阶段

##### 2.1 Query处理
```python
question = "What is the first-line treatment for type 2 diabetes?"

# NER提取
question_entities = ["treatment", "type 2 diabetes"]  # BC5CDR + HF

# 实体匹配
question_entity_embeddings = encode(["treatment", "type 2 diabetes"])
# Shape: (2, 768)

similarities = dot(entity_embeddings, question_entity_embeddings.T)
# entity_embeddings: (n_entities, 768)
# similarities: (n_entities, 2)

# 为每个问题实体找最相似的库内实体
for q_idx in range(2):
    best_match_idx = argmax(similarities[:, q_idx])
    best_match_text = entity_texts[best_match_idx]
    best_match_score = similarities[best_match_idx, q_idx]

# 结果:
seed_entities = [
    ("drug therapy", 0.82, "treatment的匹配"),  # 假设
    ("type 2 diabetes mellitus", 0.96, "精确匹配")
]

# Question embedding
question_embedding = encode(question)  # (768,)
```

##### 2.2 超图检索 🔥
```python
# Step 1: 超边语义匹配
hyperedge_scores = dot(hyperedge_embeddings, question_embedding)
# hyperedge_embeddings: (2, 768)
# question_embedding: (768,)
# 结果: [0.78, 0.45]  (hyperedge_1相关度高)

# Step 2: 应用置信度权重
hyperedge_scores[0] *= 0.65  # hyperedge_1的score
hyperedge_scores[1] *= 1.0   # hyperedge_2的score
# 结果: [0.507, 0.45]

# Step 3: Top-K筛选
sorted_indices = argsort([0.507, 0.45])[::-1]  # [0, 1]
top_hyperedges = [
    ("he_abc123", "Metformin is the first-line...", 0.507),
    ("he_def456", "It reduces hepatic glucose...", 0.45)
]

# Step 4: 双向实体扩展
expanded_entities = set()

# 方向A: 从top超边扩展
# he_abc123 → {"metformin", "type 2 diabetes mellitus"}
expanded_entities.update(["metformin", "type 2 diabetes mellitus"])
# he_def456 → {"glucose", "insulin", ...}
expanded_entities.update(["glucose", "insulin", "hepatic glucose production", "insulin sensitivity"])

# 方向B: 从种子实体扩展
# seed: "type 2 diabetes mellitus"
related_hyperedges = entity_to_hyperedges["type 2 diabetes mellitus"]
# → {"he_abc123"} (已经在top中)

# seed: "drug therapy" (假设库中还关联其他超边)
# 假设关联到其他糖尿病药物的超边,扩展更多实体...

# 最终扩展实体: 
expanded_entities = {
    "metformin", "type 2 diabetes mellitus", "glucose", "insulin",
    "hepatic glucose production", "insulin sensitivity",
    # + 其他从种子实体扩展的相关实体
}
```

##### 2.3 LinearRAG图检索
```python
# Step 1: DPR候选池
dpr_scores = dot(passage_embeddings, question_embedding)
# 假设我们的passage排在第10位: dpr_scores[10] = 0.68
candidate_passages = argsort(dpr_scores)[:500]  # Top-500
# passage_hash_id在候选池中

# Step 2: 实体扩展 (图遍历)
activated_entities = {
    "type 2 diabetes mellitus": (idx1, 0.96, tier=1),  # 种子实体
    "drug therapy": (idx2, 0.82, tier=1)
}

# Iteration 1
for entity_id, (_, score, tier) in activated_entities.items():
    # 获取包含该实体的句子
    if entity_id == "type 2 diabetes mellitus":
        sentences = [sentence_1_hash_id, ...]  # 句子:"Metformin is..."
        
        # 计算句子相似度
        sent_embeddings = sentence_embeddings[[sentence_1_idx, ...]]
        sent_similarities = dot(sent_embeddings, question_embedding)
        # [0.82, ...]
        
        # Top句子的实体
        top_sent = sentence_1
        new_entities = sentence_hash_id_to_entity_hash_ids[sentence_1_hash_id]
        # → {"metformin", "type 2 diabetes mellitus"}
        
        # 更新权重
        entity_weights[metformin_idx] += 0.96 * 0.82  # seed_score * sent_similarity
        # 新实体: metformin (tier=2)
        activated_entities["metformin"] = (metformin_idx, 0.96*0.82, 2)

# Iteration 2 (如果需要)
# 从metformin继续扩展...

# Step 3: Passage权重计算
passage_weights = zeros(n_nodes)

for passage_idx in candidate_passages:
    if passage_hash_id == current_passage_hash_id:
        passage_text = "Metformin is the first-line..."
        
        # DPR基础分数
        dpr_score_norm = min_max_normalize(0.68)  # 假设0.75
        
        # 实体匹配加成
        entity_bonus = 0
        
        # "type 2 diabetes mellitus"出现1次
        entity_bonus += 0.96 * log(1 + 1) / 1  # score * log(1+count) / tier
        # = 0.96 * 0.693 = 0.665
        
        # "metformin"出现1次
        entity_bonus += 0.787 * log(1 + 1) / 2  # tier=2
        # = 0.787 * 0.693 / 2 = 0.273
        
        # "drug therapy"不在文本中,跳过
        
        # 总加成: 0.665 + 0.273 = 0.938
        
        # 最终权重
        passage_weights[passage_node_idx] = (
            0.7 * 0.75 +          # passage_ratio * dpr_score
            log(1 + 0.938)        # log(1 + entity_bonus)
        )
        # = 0.525 + 0.661 = 1.186

# Step 4: PPR
node_weights = entity_weights + passage_weights
# node_weights[passage_node_idx] = 1.186
# node_weights[metformin_idx] = 0.787
# node_weights[type2dm_idx] = 0.96
# ...

pagerank_scores = graph.personalized_pagerank(
    reset=node_weights,
    damping=0.85,
    weights='weight'
)
# PPR从高权重节点开始随机游走,权重沿边传播
# 结果: passage_node获得高PageRank分数 (因为直接连接高权重实体)

# 提取passage排序
doc_scores = pagerank_scores[passage_node_indices]
sorted_passages = argsort(doc_scores)[::-1]
# 我们的passage很可能排在前列
```

##### 2.4 超图增强 🔥

**关键问题**: 扩展实体有上百个，但最终只给LLM Top-K(如5个)文档，如何筛选？

**答案**: 扩展实体不是直接给LLM，而是用来**重新排序所有候选passages**！

```python
# === 步骤1: PPR图检索返回排序的所有passages ===
# (注意: 这里返回的是全部排序结果，不是Top-K)
passage_hash_ids, passage_scores = graph_search_with_seed_entities(...)
# 例如: 20000个passages的完整排序
# [passage_1, passage_2, ..., passage_20000]
# [score_1,   score_2,   ..., score_20000]

# === 步骤2: 超图检索得到扩展实体 ===
expanded_entities = hypergraph_retrieve(...)
# 例如: 150个扩展实体
# {"metformin", "type 2 diabetes", "glucose", "insulin", ...}

# === 步骤3: 🔥 用扩展实体boost所有passages的分数 ===
boosted_scores = []
for passage_hash_id, base_score in zip(passage_hash_ids, passage_scores):
    passage_text = hash_id_to_text[passage_hash_id]
    
    # 统计该passage包含多少个扩展实体
    entity_matches = 0
    for entity_id in expanded_entities:
        entity_text = hypergraph_store.get_entity_text(entity_id)
        if entity_text and entity_text.lower() in passage_text.lower():
            entity_matches += 1
    
    # 应用boost (包含越多扩展实体，boost越大)
    if entity_matches > 0:
        boost = 1 + (1.2 - 1) * min(entity_matches, 3) / 3
        # 最多boost到1.2倍 (当匹配3个或以上实体时)
        boosted_score = base_score * boost
    else:
        boosted_score = base_score
    
    boosted_scores.append(boosted_score)

# === 步骤4: 重新排序所有passages ===
sorted_pairs = sorted(
    zip(passage_hash_ids, boosted_scores),
    key=lambda x: x[1],
    reverse=True
)
reranked_passage_hash_ids = [p[0] for p in sorted_pairs]
reranked_passage_scores = [p[1] for p in sorted_pairs]

# === 步骤5: 🎯 最终Top-K筛选 ===
final_passage_hash_ids = reranked_passage_hash_ids[:5]  # retrieval_top_k=5
final_passage_scores = reranked_passage_scores[:5]
final_passages = [hash_id_to_text[pid] for pid in final_passage_hash_ids]

# 结果: 只有5个passages进入prompt
# 这5个是经过超图实体扩展增强后排名最高的
```

**示例说明**:

假设我们的目标passage初始排名第8:
```
初始排名 (PPR):
1. passage_A (score=1.85, 包含0个扩展实体)
2. passage_B (score=1.72, 包含1个扩展实体)
...
8. passage_target (score=1.45, 包含3个扩展实体: metformin, type 2 diabetes, insulin)
...

应用超图boost后:
- passage_A: 1.85 * 1.0 = 1.85 (无boost)
- passage_B: 1.72 * 1.067 = 1.835 (boost=1+(1.2-1)*1/3)
- passage_target: 1.45 * 1.2 = 1.74 (boost=1+(1.2-1)*3/3)

重新排名后:
1. passage_A (score=1.85)
2. passage_B (score=1.835)
3. passage_target (score=1.74) ← 从第8升至第3！
...

最终Top-5:
passage_target成功进入Top-5，会被送给LLM
```

**核心机制**:
- 扩展实体 = **信号**: 标记哪些passage更相关
- Boost = **重排序**: 提升包含扩展实体的passages
- Top-K = **截断**: 最终只取boost后排名最高的K个

##### 2.5 构建LLM上下文
```python
# 超边上下文
hyperedge_context = """[Medical Knowledge Facts]
1. Metformin is the first-line pharmacological treatment for type 2 diabetes mellitus.
2. It reduces hepatic glucose production and improves insulin sensitivity.
"""

# 完整prompt
prompt = f"""Context:
{hyperedge_context}

Passage 1: Metformin is the first-line pharmacological treatment for type 2 diabetes mellitus. It reduces hepatic glucose production and improves insulin sensitivity.

Passage 2: ... (其他检索到的passage)

...

Question: What is the first-line treatment for type 2 diabetes?
Options:
A. Insulin
B. Metformin
C. Sulfonylureas
D. DPP-4 inhibitors

YOUR RESPONSE MUST BE EXACTLY ONE LETTER: A, B, C, or D
"""

# LLM推理
response = llm.infer(prompt)
# 输出: "B"

# 答案解析
answer = parse_answer(response)  # "B"
```

---

### 关键机制总结

#### 1. 超图如何帮助检索?
```
问题: "What is the first-line treatment for type 2 diabetes?"

传统DPR:
  question_embedding → passage_embeddings
  ↓
  语义匹配可能错过关键信息 (如"first-line"这个重要限定)

LinearRAG (无超图):
  question → 实体["treatment", "type 2 diabetes"] → 图遍历 → passages
  ↓
  通过实体连接找到相关passages,但实体扩展可能不充分

LinearRAG + Hypergraph:
  question → 实体 + 超边语义匹配
  ↓
  超边 = 句子级关系 ("Metformin + type 2 diabetes + first-line treatment")
  ↓
  双向扩展: 
    - 超边→实体: 发现"metformin"(可能不在问题中)
    - 实体→超边: 发现相关临床知识
  ↓
  扩展实体增强passage排序
  ↓
  Top超边句子作为知识事实提示LLM
```

**优势**:
1. **语义补全**: 问题只提"treatment",超图帮助找到具体药物"metformin"
2. **关系保留**: 超边保留完整句子语境,不丢失"first-line"等关键限定词
3. **知识增强**: 医学模式识别提升临床相关超边,优先召回治疗建议类知识
4. **双向融合**: 同时利用问题语义(超边匹配)和实体结构(图遍历)

#### 2. 为什么需要图+超图混合?
```
图 (LinearRAG):
  优势: 结构化遍历,利用实体共现和文档组织
  局限: 二元边无法表达多实体复杂关系

超图 (Hypergraph):
  优势: n元关系,句子级语境,医学模式识别
  局限: 孤立的超边缺乏全局结构信息

混合 (HyperLinearRAG):
  图提供结构化检索路径
  ↓
  超图扩展实体并提供语义增强
  ↓
  相互补充,提升召回和准确率
```

#### 3. 超边置信度分数的作用
```python
# 构建时: 基于实体数量和医学模式
base_score = len(entities) / max_entity_count  # 0.0-1.0

# 医学模式增强
if {DISEASE, CHEMICAL} in entity_types:
    score *= 1.3  # 疾病-药物: 临床高相关

# 检索时: 调整超边重要性
hyperedge_score = semantic_similarity * confidence_score
# 既考虑语义匹配,也考虑医学领域价值
```

这确保了临床相关的超边(如"症状-诊断"、"疾病-治疗")在检索时获得更高权重。

---

## 性能指标

根据配置和代码注释:

| 组件 | 规模 | 说明 |
|------|------|------|
| Passages | 20000 | PubMed文献chunks |
| Entities | ~50000 | 医学实体 (CHEMICAL, DISEASE等) |
| Sentences | ~100000 | 包含实体的句子 |
| Hyperedges | ~60000 | 句子级超边(≥2实体) |
| Graph nodes | ~170000 | passages + entities + sentences |
| Graph edges | ~500000 | passage-entity + entity-sentence + passage-passage |
| Hypergraph edges | ~120000 | entity-hyperedge (二部图) |

检索性能:
- **索引时间**: ~200-300秒 (20k passages)
- **单query检索**: ~0.5-1秒
  - 超图检索: ~0.1秒
  - 图遍历(PPR): ~0.3秒
  - 增强+排序: ~0.1秒
- **准确率提升**: 相比纯DPR提升5-10% (根据MIRAGE基准)

---

## 总结

**LinearRAG + Hypergraph的核心创新**:

1. **多层次知识表示**:
   - 文档层: passage embeddings (DPR)
   - 实体层: entity graph (结构化知识)
   - 关系层: hyperedges (句子级n元关系)

2. **混合检索策略**:
   - 密集检索 (DPR): 语义相似度
   - 图遍历 (PPR): 实体连接和结构
   - 超图匹配: 多实体关系和医学模式

3. **双向融合机制**:
   - Top-down: 问题→超边→实体→passages
   - Bottom-up: 问题→实体→图遍历→passages
   - 超图扩展的实体增强最终排序

4. **领域优化**:
   - 医学NER (BC5CDR + HuggingFace)
   - 医学关系模式识别 (疾病-药物、症状-诊断等)
   - 临床相关性re-ranking

5. **🔥 扩展实体的作用机制**:
   - **不是直接输入**: 扩展实体(~150个)不会直接给LLM
   - **用于重排序**: 作为信号boost包含这些实体的passages
   - **最终截断**: 只有重排序后Top-K(5个)passages进入prompt
   - **效果**: 让原本排名较低但包含关键实体的文档得以浮现

这使得系统能够在大规模医学文献中准确找到回答复杂临床问题所需的知识片段。

---

## 完整数据流总结

```
输入: Question "What is the first-line treatment for type 2 diabetes?"
  ↓
[NER] → 种子实体: ["treatment", "type 2 diabetes"]
  ↓
[超图检索] 
  → Top-30超边 (语义匹配)
  → 扩展实体: ~150个 ["metformin", "insulin", "glucose", ...]
  ↓
[图检索PPR]
  → 排序所有20000个passages (基于实体图遍历)
  ↓
[超图增强]
  → 用150个扩展实体重排序所有passages
  → 包含更多扩展实体的passage分数提升
  ↓
[Top-K截断]
  → 取重排序后Top-5 passages
  ↓
[LLM生成]
  → Prompt = [超边上下文(Top-5超边句子)] + [Top-5 passages] + [Question]
  → 答案: "B. Metformin"
```

**关键洞察**: 
- 扩展实体是**排序信号**，不是**输入内容**
- 超边文本(Top-5)作为**知识提示**进入prompt
- Passages(Top-5)是**主要上下文**
- 三者协同，确保LLM看到最相关的信息

