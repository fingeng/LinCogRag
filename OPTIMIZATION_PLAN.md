# LinCogRAG 准确率优化方案

> 基于当前实验结果和代码分析，制定针对性优化策略

## 📊 当前状态分析

### 实验结果 (2024-12-23)

| 数据集 | 当前 | 目标(MedRAG) | 差距 | 问题类型 |
|--------|------|-------------|------|----------|
| MedQA | 81.23% | 82.80% | -1.57% | 选择题 |
| MedMCQA | 49.42% | 66.65% | -17.23% | 选择题 |
| MMLU-Med | 22.77% | 87.24% | **-64.47%** | 选择题 |
| PubMedQA | 0% | 70.60% | **-70.60%** | Yes/No/Maybe |
| BioASQ | 0% | 92.56% | **-92.56%** | Yes/No |

### 核心问题

1. **PubMedQA/BioASQ全部INVALID**: LLM答案解析完全失败
2. **MMLU准确率极低**: 检索的PubMed文献与通用医学知识不匹配
3. **超图增强效果有限**: 当前是后处理boost，未深度融合

---

## 🚀 优化方案

### Phase 1: 紧急修复 - 答案解析问题

**问题根因**: LLM返回的格式不符合预期，正则匹配失败

**修复策略**:

```python
# 1. 增强系统提示 - 更严格的格式要求
PUBMEDQA_SYSTEM_PROMPT = """You are a medical expert. Based on the provided context, determine if the statement in the question is supported.

CRITICAL: Your response must be EXACTLY one word from: Yes, No, Maybe
- Yes: if the evidence clearly supports the statement
- No: if the evidence clearly contradicts the statement  
- Maybe: if the evidence is inconclusive or insufficient

DO NOT add any explanation, punctuation, or other text.
ONLY output one word: Yes, No, or Maybe"""

# 2. 增强解析逻辑 - 多级fallback
def parse_yesno_maybe(response: str) -> str:
    text = response.strip().lower()
    
    # Level 1: 直接匹配
    if text in ['yes', 'no', 'maybe']:
        return text.capitalize()
    
    # Level 2: 正则匹配
    match = re.search(r'\b(yes|no|maybe)\b', text, re.IGNORECASE)
    if match:
        return match.group(1).capitalize()
    
    # Level 3: 语义推断
    positive_signals = ['support', 'confirm', 'true', 'correct', 'evidence shows']
    negative_signals = ['contradict', 'false', 'incorrect', 'no evidence', 'not support']
    uncertain_signals = ['uncertain', 'inconclusive', 'insufficient', 'unclear']
    
    if any(s in text for s in positive_signals):
        return 'Yes'
    if any(s in text for s in negative_signals):
        return 'No'
    if any(s in text for s in uncertain_signals):
        return 'Maybe'
    
    # Level 4: 首字母判断（如果LLM返回完整句子）
    first_word = text.split()[0] if text.split() else ''
    if first_word in ['yes', 'no', 'maybe']:
        return first_word.capitalize()
    
    return 'INVALID'
```

### Phase 2: 数据集自适应检索

**核心思想**: 不同数据集需要不同的检索策略

```python
class DatasetAdaptiveRetriever:
    """针对不同数据集特点的自适应检索"""
    
    def retrieve(self, question: str, dataset: str) -> List[str]:
        if dataset in ['medqa', 'medmcqa']:
            # 选择题：答案感知检索
            return self.option_aware_retrieve(question)
        
        elif dataset == 'pubmedqa':
            # PubMedQA：证据决定性检索
            return self.decisive_evidence_retrieve(question)
        
        elif dataset == 'bioasq':
            # BioASQ：高置信度精准检索
            return self.high_confidence_retrieve(question)
        
        elif dataset == 'mmlu':
            # MMLU：知识补充检索
            return self.knowledge_augment_retrieve(question)
    
    def option_aware_retrieve(self, question: str) -> List[str]:
        """选择题：分别检索每个选项的支持证据"""
        # 提取选项
        options = self.extract_options(question)
        
        # 为每个选项找支持证据
        option_evidence = {}
        for opt_key, opt_text in options.items():
            evidence = self.retrieve_for_option(question, opt_text)
            option_evidence[opt_key] = evidence
        
        # 返回证据最强的选项的passages
        best_option = max(option_evidence, key=lambda k: self.score_evidence(option_evidence[k]))
        return option_evidence[best_option]
    
    def decisive_evidence_retrieve(self, question: str) -> List[str]:
        """PubMedQA：寻找决定性证据"""
        candidates = self.base_retrieve(question, top_k=30)
        
        # 评估每个passage的决定性
        scored = []
        for passage in candidates:
            decisiveness = self.score_decisiveness(passage)
            direction = self.evidence_direction(passage)  # positive/negative/neutral
            scored.append((passage, decisiveness, direction))
        
        # 选择决定性最强且方向一致的证据
        scored.sort(key=lambda x: x[1], reverse=True)
        
        # 检查证据一致性
        positive = [p for p, d, dir in scored if dir == 'positive' and d > 0.5]
        negative = [p for p, d, dir in scored if dir == 'negative' and d > 0.5]
        
        if len(positive) >= 3 and len(negative) <= 1:
            return positive[:3]  # 强正向证据
        elif len(negative) >= 3 and len(positive) <= 1:
            return negative[:3]  # 强负向证据
        else:
            return [p for p, d, dir in scored[:3]]  # 证据冲突，返回最相关的
```

### Phase 3: 证据决定性评分

```python
class EvidenceDecisivenessScorer:
    """评估文档对答案的决定性贡献"""
    
    DECISIVE_POSITIVE = [
        "clearly", "definitively", "proven", "established",
        "strongly associated", "significant effect", "recommended",
        "first-line", "gold standard", "treatment of choice"
    ]
    
    DECISIVE_NEGATIVE = [
        "no evidence", "not recommended", "contraindicated",
        "ineffective", "no significant", "failed to show"
    ]
    
    UNCERTAIN_INDICATORS = [
        "may", "might", "possibly", "unclear", "conflicting",
        "limited evidence", "further research"
    ]
    
    def score_decisiveness(self, passage: str) -> float:
        """计算决定性分数 (0-1)"""
        text = passage.lower()
        
        pos = sum(1 for i in self.DECISIVE_POSITIVE if i in text)
        neg = sum(1 for i in self.DECISIVE_NEGATIVE if i in text)
        unc = sum(1 for i in self.UNCERTAIN_INDICATORS if i in text)
        
        total = pos + neg + unc
        if total == 0:
            return 0.5
        
        return (pos + neg) / total
    
    def evidence_direction(self, passage: str) -> str:
        """判断证据方向"""
        text = passage.lower()
        
        pos = sum(1 for i in self.DECISIVE_POSITIVE if i in text)
        neg = sum(1 for i in self.DECISIVE_NEGATIVE if i in text)
        
        if pos > neg + 1:
            return 'positive'
        elif neg > pos + 1:
            return 'negative'
        return 'neutral'
```

### Phase 4: 超图深度融合

当前问题: 超图信息仅用于后处理boost，未参与PPR计算

**改进方案**: 将超图信息预编码到PPR重启分布

```python
def hypergraph_enhanced_ppr(self, question_embedding, seed_entities):
    """超图增强的统一PPR"""
    
    # 1. 计算基础实体权重
    entity_weights = self.calculate_entity_scores(seed_entities)
    
    # 2. 🔥 超图感知的实体扩展
    for seed_entity_id, seed_score in seed_entities:
        # 找到包含该实体的高分超边
        related_hyperedges = self.hypergraph_store.get_hyperedges_by_entity(seed_entity_id)
        
        for he_id in related_hyperedges:
            he_score = self.hypergraph_store.get_hyperedge_score(he_id)
            if he_score < 0.5:
                continue
            
            # 超边中的共现实体获得传播权重
            co_entities = self.hypergraph_store.get_entities_by_hyperedge(he_id)
            for co_entity_id in co_entities:
                if co_entity_id in self.node_name_to_vertex_idx:
                    co_idx = self.node_name_to_vertex_idx[co_entity_id]
                    # 传播权重 = 种子分数 × 超边分数 × 衰减
                    propagated = seed_score * he_score * 0.5
                    entity_weights[co_idx] += propagated
    
    # 3. 统一PPR (已融合超图信息)
    passage_weights = self.calculate_passage_scores(entity_weights)
    return self.run_ppr(entity_weights + passage_weights)
```

### Phase 5: MMLU知识增强

**问题**: MMLU是通用医学知识题，PubMed临床文献可能不直接相关

**解决方案**:

```python
def mmlu_knowledge_retrieve(self, question: str) -> List[str]:
    """MMLU专用：知识概念匹配"""
    
    # 1. 提取问题中的关键医学概念
    concepts = self.extract_medical_concepts(question)
    
    # 2. 扩展查询：添加医学同义词
    expanded_query = self.expand_with_synonyms(question, concepts)
    
    # 3. 检索时优先匹配概念定义类passage
    candidates = self.retrieve_with_concept_priority(expanded_query)
    
    # 4. 如果检索质量低，回退到LLM内部知识
    if self.assess_retrieval_quality(candidates, question) < 0.3:
        return []  # 不提供context，让LLM用内部知识回答
    
    return candidates[:3]
```

---

## 📋 实施计划

### 优先级排序

| 优先级 | 任务 | 预期收益 | 实现难度 |
|--------|------|----------|----------|
| P0 | 修复答案解析 | PubMedQA/BioASQ从0%恢复 | 简单 |
| P1 | 数据集自适应检索 | 整体+5-10% | 中等 |
| P2 | 证据决定性评分 | PubMedQA +3-5% | 简单 |
| P3 | 超图深度融合 | 整体+2-3% | 中等 |
| P4 | MMLU知识增强 | MMLU +10-20% | 中等 |

### 预期结果

优化后目标:

| 数据集 | 当前 | 优化后 | 目标 |
|--------|------|--------|------|
| MedQA | 81.23% | 83% | 82.80% |
| MedMCQA | 49.42% | 60% | 66.65% |
| MMLU-Med | 22.77% | 50% | 87.24% |
| PubMedQA | 0% | 65% | 70.60% |
| BioASQ | 0% | 85% | 92.56% |
| **平均** | **30.68%** | **68.6%** | **79.97%** |

---

## 🔧 代码修改清单

1. `src/LinearRAG.py`:
   - 增强qa()方法中的系统提示
   - 改进答案解析逻辑
   - 添加数据集自适应检索
   - 实现超图深度融合

2. 新增 `src/retrieval_strategies.py`:
   - EvidenceDecisivenessScorer
   - DatasetAdaptiveRetriever
   - OptionAwareRetriever

3. 修改 `src/config.py`:
   - 添加数据集特定配置参数

4. 修改 `experiments/run_lincog_benchmark.py`:
   - 添加详细日志记录
   - 保存原始LLM响应用于调试
