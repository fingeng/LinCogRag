# 巧妙提升准确率策略

> 从算法本质出发，用智慧而非蛮力提升性能

## 📋 目录
1. [问题本质分析](#1-问题本质分析)
2. [策略一：证据决定性评分](#2-策略一证据决定性评分)
3. [策略二：答案感知检索](#3-策略二答案感知检索)
4. [策略三：超图条件完备性](#4-策略三超图条件完备性)
5. [策略四：自适应精准检索](#5-策略四自适应精准检索)
6. [策略五：对比式选项匹配](#6-策略五对比式选项匹配)
7. [策略六：多粒度证据聚焦](#7-策略六多粒度证据聚焦)
8. [实施建议](#8-实施建议)

---

## 1. 问题本质分析

### 1.1 为什么"更多"可能导致"更差"？

```
场景：PubMedQA "Does aspirin prevent heart attacks?"

Top-5检索结果（当前）:
  Doc1: "Aspirin reduces cardiovascular events..." → 支持Yes
  Doc2: "Low-dose aspirin shows benefits..."      → 支持Yes
  Doc3: "Aspirin has antiplatelet effects..."     → 中立
  
LLM判断: Yes ✅

Top-15检索结果（暴力增加）:
  Doc1-3: 同上
  Doc4: "Aspirin may cause bleeding risks..."     → 支持No
  Doc5: "Evidence is mixed for primary prevention" → 支持Maybe
  Doc6: "Some studies show no benefit..."         → 支持No
  ...
  
LLM判断: Maybe ❌ (信息冲突，倾向保守)
```

**核心问题**：不是检索数量，而是**证据的决定性**。

### 1.2 三类问题的本质差异

| 问题类型 | 特点 | 需要的证据 |
|---------|------|-----------|
| 选择题(MedQA等) | 有明确正确答案 | 能区分选项的关键信息 |
| Yes/No(BioASQ) | 需要明确判断 | 决定性证据，避免模糊信息 |
| Yes/No/Maybe(PubMedQA) | 允许不确定 | 需要判断证据充分性 |

### 1.3 新思路：从"相关性检索"到"决定性检索"

```
传统RAG: 检索最相关的文档
智慧RAG: 检索能决定答案的证据
```

---

## 2. 策略一：证据决定性评分 (Evidence Decisiveness Scoring)

### 2.1 核心思想

不是所有相关文档都有价值，只有能**明确支持/反驳某个答案**的文档才有决定性。

### 2.2 决定性评分算法

```python
class EvidenceDecisivenessScorer:
    """评估文档对答案决定性的贡献"""
    
    # 决定性指示词
    DECISIVE_POSITIVE = [
        "clearly", "definitively", "proven", "established",
        "strongly associated", "significant effect", "recommended",
        "first-line", "gold standard", "treatment of choice"
    ]
    
    DECISIVE_NEGATIVE = [
        "no evidence", "not recommended", "contraindicated",
        "ineffective", "no significant", "failed to show",
        "should not", "avoid"
    ]
    
    UNCERTAIN_INDICATORS = [
        "may", "might", "possibly", "unclear", "conflicting",
        "limited evidence", "further research", "controversial",
        "some studies", "mixed results"
    ]
    
    def score_decisiveness(self, passage: str, question: str) -> Tuple[float, str]:
        """
        返回: (决定性分数, 倾向方向)
        分数范围: 0-1, 越高越决定性
        方向: "positive", "negative", "uncertain"
        """
        passage_lower = passage.lower()
        
        # 计算各类指示词出现次数
        positive_count = sum(1 for ind in self.DECISIVE_POSITIVE if ind in passage_lower)
        negative_count = sum(1 for ind in self.DECISIVE_NEGATIVE if ind in passage_lower)
        uncertain_count = sum(1 for ind in self.UNCERTAIN_INDICATORS if ind in passage_lower)
        
        # 决定性分数 = (正面+负面指示) / (正面+负面+不确定指示)
        total = positive_count + negative_count + uncertain_count
        if total == 0:
            return 0.5, "neutral"
        
        decisiveness = (positive_count + negative_count) / total
        
        # 确定倾向方向
        if positive_count > negative_count:
            direction = "positive"
        elif negative_count > positive_count:
            direction = "negative"
        else:
            direction = "neutral"
        
        return decisiveness, direction
    
    def filter_decisive_passages(self, passages: List[str], question: str, 
                                  min_decisiveness: float = 0.6) -> List[str]:
        """只保留决定性强的文档"""
        scored = []
        for p in passages:
            decisiveness, direction = self.score_decisiveness(p, question)
            if decisiveness >= min_decisiveness:
                scored.append((p, decisiveness, direction))
        
        # 按决定性排序
        scored.sort(key=lambda x: x[1], reverse=True)
        return [p[0] for p in scored]
```

### 2.3 融入检索流程

```python
def smart_retrieve(self, question, question_embedding, seed_entities):
    """智慧检索：优先决定性证据"""
    
    # 1. 常规检索获取候选
    candidates = self.standard_retrieve(question_embedding, top_n=50)
    
    # 2. 决定性评分过滤
    scorer = EvidenceDecisivenessScorer()
    decisive_passages = []
    
    for passage_id in candidates:
        passage_text = self.get_passage_text(passage_id)
        decisiveness, direction = scorer.score_decisiveness(passage_text, question)
        decisive_passages.append((passage_id, passage_text, decisiveness, direction))
    
    # 3. 智能选择策略
    # 优先选择决定性强且方向一致的文档
    decisive_passages.sort(key=lambda x: x[2], reverse=True)
    
    # 检查证据一致性
    positive_evidence = [p for p in decisive_passages if p[3] == "positive"]
    negative_evidence = [p for p in decisive_passages if p[3] == "negative"]
    
    # 如果证据方向明确一致，只取该方向的top文档
    if len(positive_evidence) >= 3 and len(negative_evidence) <= 1:
        final = positive_evidence[:5]
    elif len(negative_evidence) >= 3 and len(positive_evidence) <= 1:
        final = negative_evidence[:5]
    else:
        # 证据冲突时，取最决定性的几个
        final = decisive_passages[:5]
    
    return [p[0] for p in final]
```

---

## 3. 策略二：答案感知检索 (Answer-Aware Retrieval)

### 3.1 核心思想

传统RAG只用问题检索，忽略了选项信息。对于选择题，**选项本身是强信号**。

### 3.2 两阶段检索

```
阶段1: Question-based检索 → 候选文档池
阶段2: Option-based精筛 → 每个选项的支持证据
最终: 选择证据最强的选项
```

### 3.3 实现算法

```python
class AnswerAwareRetriever:
    """答案感知的检索器"""
    
    def retrieve_for_mcq(self, question: str, options: List[str], 
                         question_embedding, top_k: int = 5):
        """
        为选择题检索，返回最可能的答案和支持证据
        """
        
        # 阶段1: 问题检索获取候选池
        candidate_passages = self.base_retrieve(question_embedding, top_n=100)
        
        # 阶段2: 计算每个选项的证据支持度
        option_scores = {}
        option_evidence = {}
        
        for option_idx, option_text in enumerate(options):
            option_letter = chr(65 + option_idx)  # A, B, C, D
            
            # 方法1: 选项与候选文档的语义相似度
            option_embedding = self.encode(option_text)
            
            # 方法2: 选项关键词在文档中的匹配
            option_keywords = self.extract_keywords(option_text)
            
            evidence_scores = []
            for passage_id in candidate_passages:
                passage_text = self.get_passage_text(passage_id)
                passage_embedding = self.get_passage_embedding(passage_id)
                
                # 语义相似度
                semantic_sim = np.dot(option_embedding, passage_embedding)
                
                # 关键词匹配度
                keyword_matches = sum(1 for kw in option_keywords 
                                     if kw.lower() in passage_text.lower())
                keyword_score = keyword_matches / len(option_keywords) if option_keywords else 0
                
                # 综合分数
                combined = 0.6 * semantic_sim + 0.4 * keyword_score
                evidence_scores.append((passage_id, combined))
            
            # 取该选项的top证据
            evidence_scores.sort(key=lambda x: x[1], reverse=True)
            top_evidence = evidence_scores[:3]
            
            option_scores[option_letter] = sum(e[1] for e in top_evidence)
            option_evidence[option_letter] = [e[0] for e in top_evidence]
        
        # 选择证据最强的选项
        best_option = max(option_scores, key=option_scores.get)
        
        # 返回该选项的证据
        return option_evidence[best_option], best_option, option_scores
    
    def retrieve_for_yesno(self, question: str, question_embedding, top_k: int = 5):
        """
        为Yes/No问题检索，分别找支持和反对的证据
        """
        
        # 候选池
        candidates = self.base_retrieve(question_embedding, top_n=50)
        
        # 构造Yes/No的查询变体
        yes_query = question.replace("?", "") + " Yes, this is true."
        no_query = question.replace("?", "") + " No, this is false."
        
        yes_embedding = self.encode(yes_query)
        no_embedding = self.encode(no_query)
        
        yes_evidence = []
        no_evidence = []
        
        for passage_id in candidates:
            passage_embedding = self.get_passage_embedding(passage_id)
            
            yes_sim = np.dot(yes_embedding, passage_embedding)
            no_sim = np.dot(no_embedding, passage_embedding)
            
            if yes_sim > no_sim + 0.1:  # 明显支持Yes
                yes_evidence.append((passage_id, yes_sim))
            elif no_sim > yes_sim + 0.1:  # 明显支持No
                no_evidence.append((passage_id, no_sim))
        
        # 根据证据强度决定答案
        yes_score = sum(e[1] for e in yes_evidence[:5])
        no_score = sum(e[1] for e in no_evidence[:5])
        
        if yes_score > no_score * 1.2:
            return [e[0] for e in yes_evidence[:top_k]], "Yes"
        elif no_score > yes_score * 1.2:
            return [e[0] for e in no_evidence[:top_k]], "No"
        else:
            # 证据不够决定性，返回混合证据
            return candidates[:top_k], "Maybe"
```

### 3.4 关键洞察

```
传统: Question → Retrieve → LLM(Question + Docs) → Answer
改进: Question + Options → Retrieve per Option → Compare Evidence → Answer

区别: 让检索阶段就参与"选择"，而非把所有文档堆给LLM判断
```

---

## 4. 策略三：超图条件完备性 (Hypergraph Condition Completeness)

### 4.1 核心思想

医学推理往往需要**多个条件同时满足**。超图的n元关系正好可以检查这种"条件完备性"。

### 4.2 例子说明

```
问题: "Type 2 diabetes患者，HbA1c 9%，无肾病，首选什么药物？"

需要满足的条件:
- 条件1: Type 2 diabetes
- 条件2: HbA1c高(9%)
- 条件3: 无肾功能问题

超边示例:
超边A: "Metformin is first-line for type 2 diabetes with HbA1c > 7%"
  → 包含实体: [metformin, type 2 diabetes, HbA1c]
  → 满足条件1,2 ✓

超边B: "Metformin is contraindicated in severe renal impairment"
  → 包含实体: [metformin, renal impairment]
  → 与条件3相关，但问题说无肾病

最佳答案需要同时匹配最多条件的超边！
```

### 4.3 条件完备性算法

```python
class HypergraphConditionChecker:
    """基于超图检查答案的条件完备性"""
    
    def check_condition_completeness(self, question_entities: List[str], 
                                      hyperedges: List[Hyperedge]) -> List[Tuple[Hyperedge, float]]:
        """
        检查超边对问题条件的覆盖程度
        
        返回: (超边, 完备性分数) 列表
        """
        scored_hyperedges = []
        
        for he in hyperedges:
            # 计算超边实体与问题实体的重叠
            he_entities = set(e.lower() for e in he.entities)
            q_entities = set(e.lower() for e in question_entities)
            
            # 方法1: Jaccard相似度
            intersection = len(he_entities & q_entities)
            union = len(he_entities | q_entities)
            jaccard = intersection / union if union > 0 else 0
            
            # 方法2: 问题条件覆盖率 (更重要!)
            # 问题中有多少条件被超边覆盖
            coverage = intersection / len(q_entities) if q_entities else 0
            
            # 方法3: 超边精确度
            # 超边中有多少实体是问题相关的（避免噪声实体）
            precision = intersection / len(he_entities) if he_entities else 0
            
            # 综合分数：覆盖率为主，精确度为辅
            completeness = 0.7 * coverage + 0.3 * precision
            
            # 医学模式加成
            completeness *= he.score
            
            scored_hyperedges.append((he, completeness))
        
        # 按完备性排序
        scored_hyperedges.sort(key=lambda x: x[1], reverse=True)
        return scored_hyperedges
    
    def select_complete_evidence(self, question: str, question_entities: List[str],
                                  all_hyperedges: List[Hyperedge], top_k: int = 5):
        """
        选择条件最完备的超边作为证据
        """
        
        # 1. 检查所有超边的条件完备性
        scored = self.check_condition_completeness(question_entities, all_hyperedges)
        
        # 2. 贪心选择：最大化条件覆盖
        selected = []
        covered_conditions = set()
        
        for he, score in scored:
            if len(selected) >= top_k:
                break
            
            # 检查这个超边是否带来新的条件覆盖
            he_entities = set(e.lower() for e in he.entities)
            new_coverage = he_entities - covered_conditions
            
            if new_coverage or score > 0.8:  # 高分超边总是保留
                selected.append(he)
                covered_conditions.update(he_entities)
        
        # 3. 检查覆盖完整性
        q_entities = set(e.lower() for e in question_entities)
        final_coverage = len(covered_conditions & q_entities) / len(q_entities)
        
        return selected, final_coverage
```

### 4.4 融入检索流程

```python
def hypergraph_completeness_retrieve(self, question, question_embedding, seed_entities):
    """基于条件完备性的超图检索"""
    
    # 1. 提取问题中的所有条件实体
    question_entities = self.extract_all_entities(question)
    
    # 2. 找到高完备性的超边
    checker = HypergraphConditionChecker()
    complete_hyperedges, coverage = checker.select_complete_evidence(
        question, question_entities, self.all_hyperedges, top_k=10
    )
    
    # 3. 从高完备性超边找对应的Passage
    relevant_passages = set()
    for he in complete_hyperedges:
        passages = self.hyperedge_to_passages.get(he.hash_id, [])
        relevant_passages.update(passages)
    
    # 4. 如果覆盖率不够，用传统方法补充
    if coverage < 0.7:
        additional = self.standard_retrieve(question_embedding, top_n=20)
        for p in additional:
            if len(relevant_passages) >= 10:
                break
            relevant_passages.add(p)
    
    return list(relevant_passages)[:5], coverage
```

---

## 5. 策略四：自适应精准检索 (Adaptive Precision Retrieval)

### 5.1 核心思想

不同问题需要不同的检索策略：
- 简单问题：1-2个精确文档就够
- 复杂问题：需要多个文档互补
- 模糊问题：应该直接回答Maybe而非强行检索

### 5.2 问题复杂度评估

```python
class QuestionComplexityAnalyzer:
    """分析问题复杂度，决定检索策略"""
    
    def analyze(self, question: str, entities: List[str]) -> Dict:
        """
        返回问题分析结果
        """
        analysis = {
            "complexity": "simple",  # simple, medium, complex
            "evidence_type": "single",  # single, multiple, comparison
            "confidence_threshold": 0.7,
            "recommended_top_k": 3,
        }
        
        question_lower = question.lower()
        
        # 复杂度指标1: 实体数量
        entity_count = len(entities)
        
        # 复杂度指标2: 问题长度
        word_count = len(question.split())
        
        # 复杂度指标3: 特殊结构
        has_comparison = any(w in question_lower for w in 
                            ["compared to", "versus", "vs", "difference", "better"])
        has_multiple_conditions = "and" in question_lower or "with" in question_lower
        has_negation = any(w in question_lower for w in 
                          ["not", "except", "contraindicated", "avoid"])
        has_temporal = any(w in question_lower for w in 
                          ["first", "initial", "before", "after", "duration"])
        
        # 判断复杂度
        complexity_score = (
            entity_count * 0.3 +
            (word_count / 20) * 0.2 +
            has_comparison * 0.2 +
            has_multiple_conditions * 0.15 +
            has_negation * 0.1 +
            has_temporal * 0.05
        )
        
        if complexity_score < 0.4:
            analysis["complexity"] = "simple"
            analysis["recommended_top_k"] = 2  # 简单问题少检索
            analysis["confidence_threshold"] = 0.8  # 要求高置信度
        elif complexity_score < 0.7:
            analysis["complexity"] = "medium"
            analysis["recommended_top_k"] = 4
            analysis["confidence_threshold"] = 0.6
        else:
            analysis["complexity"] = "complex"
            analysis["recommended_top_k"] = 6
            analysis["confidence_threshold"] = 0.5
        
        # 证据类型
        if has_comparison:
            analysis["evidence_type"] = "comparison"
        elif has_multiple_conditions:
            analysis["evidence_type"] = "multiple"
        else:
            analysis["evidence_type"] = "single"
        
        return analysis
```

### 5.3 自适应检索策略

```python
def adaptive_retrieve(self, question, question_embedding, seed_entities):
    """根据问题复杂度自适应调整检索策略"""
    
    # 1. 分析问题复杂度
    analyzer = QuestionComplexityAnalyzer()
    analysis = analyzer.analyze(question, seed_entities)
    
    # 2. 根据复杂度选择策略
    if analysis["complexity"] == "simple":
        # 简单问题：精准检索，少量高质量文档
        passages = self.precision_retrieve(
            question_embedding, 
            top_k=analysis["recommended_top_k"],
            threshold=analysis["confidence_threshold"]
        )
        
    elif analysis["complexity"] == "medium":
        # 中等问题：平衡检索
        passages = self.balanced_retrieve(
            question_embedding, seed_entities,
            top_k=analysis["recommended_top_k"]
        )
        
    else:  # complex
        # 复杂问题：多角度检索
        if analysis["evidence_type"] == "comparison":
            # 对比型问题：分别检索两方面证据
            passages = self.comparison_retrieve(question, question_embedding)
        else:
            # 多条件问题：使用超图条件完备性
            passages, _ = self.hypergraph_completeness_retrieve(
                question, question_embedding, seed_entities
            )
    
    return passages, analysis

def precision_retrieve(self, question_embedding, top_k=2, threshold=0.8):
    """精准检索：只返回高置信度文档"""
    
    # DPR检索
    all_scores = np.dot(self.passage_embeddings, question_embedding)
    
    # 只取超过阈值的
    high_conf_indices = np.where(all_scores > threshold)[0]
    
    if len(high_conf_indices) == 0:
        # 没有高置信度文档，取最高的几个
        top_indices = np.argsort(all_scores)[::-1][:top_k]
    else:
        # 在高置信度中取top
        high_conf_scores = all_scores[high_conf_indices]
        sorted_idx = np.argsort(high_conf_scores)[::-1][:top_k]
        top_indices = high_conf_indices[sorted_idx]
    
    return [self.passage_hash_ids[i] for i in top_indices]
```

---

## 6. 策略五：对比式选项匹配 (Contrastive Option Matching)

### 6.1 核心思想

对于选择题，关键不是"哪个文档相关"，而是"哪个选项能被文档支持"。

### 6.2 对比学习框架

```python
class ContrastiveOptionMatcher:
    """对比式选项匹配器"""
    
    def match_options(self, question: str, options: List[str], 
                      passages: List[str]) -> Tuple[str, Dict[str, float]]:
        """
        通过对比学习匹配最佳选项
        """
        
        # 1. 构造每个选项的完整陈述
        option_statements = []
        for i, opt in enumerate(options):
            # 将选项补全为完整陈述
            if "?" in question:
                statement = question.replace("?", "").strip() + " " + opt
            else:
                statement = question + " " + opt
            option_statements.append(statement)
        
        # 2. 编码
        option_embeddings = self.encode(option_statements)  # (4, 768)
        passage_embeddings = self.encode(passages)  # (n, 768)
        
        # 3. 计算每个选项与文档集的匹配度
        option_scores = {}
        for i, (opt_emb, opt_text) in enumerate(zip(option_embeddings, options)):
            option_letter = chr(65 + i)
            
            # 方法1: 最大相似度 (最强支持证据)
            max_sim = np.max(np.dot(passage_embeddings, opt_emb))
            
            # 方法2: 平均相似度 (整体支持度)
            avg_sim = np.mean(np.dot(passage_embeddings, opt_emb))
            
            # 方法3: Top-3平均 (稳健估计)
            all_sims = np.dot(passage_embeddings, opt_emb)
            top3_avg = np.mean(np.sort(all_sims)[::-1][:3])
            
            # 综合分数
            option_scores[option_letter] = 0.4 * max_sim + 0.3 * top3_avg + 0.3 * avg_sim
        
        # 4. 选择得分最高的选项
        best_option = max(option_scores, key=option_scores.get)
        
        # 5. 计算置信度 (最高分与第二高分的差距)
        sorted_scores = sorted(option_scores.values(), reverse=True)
        confidence = sorted_scores[0] - sorted_scores[1] if len(sorted_scores) > 1 else 1.0
        
        return best_option, option_scores, confidence
```

### 6.3 结合LLM的两阶段决策

```python
def two_stage_decision(self, question, options, retrieved_passages):
    """
    两阶段决策：
    1. 对比匹配预判
    2. LLM验证
    """
    
    # 阶段1: 对比式预判
    matcher = ContrastiveOptionMatcher()
    predicted_option, scores, confidence = matcher.match_options(
        question, options, retrieved_passages
    )
    
    # 阶段2: 根据置信度决定策略
    if confidence > 0.15:  # 高置信度
        # 只给LLM看支持预判选项的证据
        supporting_passages = self.filter_supporting_passages(
            retrieved_passages, options[ord(predicted_option) - 65]
        )
        prompt = self.build_focused_prompt(question, options, supporting_passages, predicted_option)
    else:
        # 低置信度，给LLM看所有证据让它判断
        prompt = self.build_full_prompt(question, options, retrieved_passages)
    
    # LLM最终决策
    answer = self.llm.infer(prompt)
    
    return answer, predicted_option, confidence
```

---

## 7. 策略六：多粒度证据聚焦 (Multi-Granularity Evidence Focus)

### 7.1 核心思想

不同问题需要不同粒度的证据：
- 有些问题需要**段落级**证据（完整上下文）
- 有些问题只需要**句子级**证据（精确信息）
- 有些问题需要**短语级**证据（关键定义）

### 7.2 多粒度检索

```python
class MultiGranularityRetriever:
    """多粒度证据检索器"""
    
    def retrieve_multi_granular(self, question, question_embedding, 
                                 seed_entities) -> Dict[str, List[str]]:
        """
        同时检索不同粒度的证据
        """
        
        # 粒度1: 段落级 (Passage)
        passage_results = self.passage_retrieve(question_embedding, top_k=10)
        
        # 粒度2: 句子级 (Sentence) - 利用已有的句子索引
        sentence_results = self.sentence_retrieve(question_embedding, seed_entities, top_k=10)
        
        # 粒度3: 超边级 (Hyperedge) - 实体关系片段
        hyperedge_results = self.hyperedge_retrieve(question_embedding, seed_entities, top_k=10)
        
        return {
            "passages": passage_results,
            "sentences": sentence_results,
            "hyperedges": hyperedge_results
        }
    
    def smart_combine(self, multi_results: Dict, question: str) -> List[str]:
        """
        智能组合不同粒度的证据
        """
        
        # 分析问题需要的粒度
        question_lower = question.lower()
        
        # 定义型问题：需要精确句子
        is_definition = any(w in question_lower for w in 
                           ["what is", "define", "meaning of", "refers to"])
        
        # 机制型问题：需要完整段落
        is_mechanism = any(w in question_lower for w in 
                          ["how does", "mechanism", "pathway", "process"])
        
        # 关系型问题：需要超边(多实体关系)
        is_relational = any(w in question_lower for w in 
                           ["relationship", "associated with", "causes", "leads to"])
        
        combined = []
        
        if is_definition:
            # 优先句子级
            combined.extend(multi_results["sentences"][:3])
            combined.extend(multi_results["hyperedges"][:2])
        elif is_mechanism:
            # 优先段落级
            combined.extend(multi_results["passages"][:3])
            combined.extend(multi_results["sentences"][:2])
        elif is_relational:
            # 优先超边级
            combined.extend(multi_results["hyperedges"][:3])
            combined.extend(multi_results["passages"][:2])
        else:
            # 平衡组合
            combined.extend(multi_results["passages"][:2])
            combined.extend(multi_results["sentences"][:2])
            combined.extend(multi_results["hyperedges"][:1])
        
        return combined
```

### 7.3 句子级直接检索（利用已有索引）

```python
def sentence_retrieve(self, question_embedding, seed_entities, top_k=10):
    """
    直接检索相关句子（比检索段落再提取更精准）
    """
    
    # 1. 语义相似度检索
    sentence_scores = np.dot(self.sentence_embeddings, question_embedding)
    
    # 2. 实体匹配加成
    for entity_id in seed_entities:
        if entity_id in self.entity_to_sentences:
            for sent_id in self.entity_to_sentences[entity_id]:
                sent_idx = self.sentence_hash_to_idx.get(sent_id)
                if sent_idx is not None:
                    sentence_scores[sent_idx] += 0.2  # 实体匹配加成
    
    # 3. 排序
    top_indices = np.argsort(sentence_scores)[::-1][:top_k]
    return [self.sentence_hash_ids[i] for i in top_indices]
```

---

## 8. 实施建议

### 8.1 优先级排序（按预期效果/实现难度）

| 优先级 | 策略 | 预期效果 | 实现难度 | 建议 |
|--------|------|---------|---------|------|
| ⭐⭐⭐⭐⭐ | 答案感知检索 | +3-5% | 中等 | **首选** |
| ⭐⭐⭐⭐ | 证据决定性评分 | +2-4% | 简单 | **首选** |
| ⭐⭐⭐⭐ | 自适应精准检索 | +2-3% | 简单 | 易实现 |
| ⭐⭐⭐ | 超图条件完备性 | +1-2% | 中等 | 利用现有结构 |
| ⭐⭐⭐ | 对比式选项匹配 | +1-2% | 中等 | 选择题专用 |
| ⭐⭐ | 多粒度证据聚焦 | +1% | 简单 | 辅助优化 |

### 8.2 推荐组合方案

```python
# 最优组合：针对不同数据集的差异化策略

def smart_qa_pipeline(question_info):
    dataset = question_info["dataset"]
    question = question_info["question"]
    
    if dataset in ["medqa", "medmcqa", "mmlu"]:
        # 选择题：答案感知 + 对比匹配
        return answer_aware_mcq_pipeline(question, question_info["options"])
    
    elif dataset == "bioasq":
        # Yes/No：决定性评分 + 精准检索
        return decisive_yesno_pipeline(question)
    
    elif dataset == "pubmedqa":
        # Yes/No/Maybe：自适应检索 + 证据一致性检查
        return adaptive_pubmedqa_pipeline(question)
```

### 8.3 快速验证方案

建议先实现**证据决定性评分**，这是最简单且效果明显的改进：

```python
# 在retrieve方法中添加决定性过滤
def retrieve_with_decisiveness(self, ...):
    # 1. 常规检索
    candidates = self.standard_retrieve(top_n=30)
    
    # 2. 决定性评分过滤
    scored = []
    for p in candidates:
        text = self.get_text(p)
        decisiveness = self.score_decisiveness(text)
        scored.append((p, decisiveness))
    
    # 3. 只保留决定性高的
    scored.sort(key=lambda x: x[1], reverse=True)
    decisive = [p for p, d in scored if d > 0.5][:5]
    
    return decisive if decisive else candidates[:5]
```

---

## 9. 关键洞察总结

### 从"更多"到"更准"

| 暴力方法 | 智慧方法 |
|---------|---------|
| 增加Top-K | 选择决定性文档 |
| 融合更多检索器 | 融合不同粒度证据 |
| 检索更多实体 | 检查条件完备性 |
| 给LLM更多上下文 | 给LLM更聚焦的证据 |

### 核心原则

1. **质量 > 数量**：3个决定性文档 > 10个相关文档
2. **精准 > 召回**：宁可漏掉相关文档，不要引入噪声
3. **聚焦 > 全面**：针对答案检索，而非泛泛检索
4. **自适应 > 固定**：不同问题用不同策略

### 对PubMedQA的特殊处理

```python
def pubmedqa_smart_strategy(question, candidates):
    """
    PubMedQA的核心问题：什么时候该答Maybe？
    
    策略：
    1. 如果证据强烈一致 → Yes/No
    2. 如果证据明显冲突 → Maybe
    3. 如果证据不足 → Maybe
    """
    
    # 评估证据
    positive, negative, neutral = classify_evidence(candidates)
    
    # 决策逻辑
    if len(positive) >= 3 and len(negative) == 0:
        return "Yes", positive[:3]
    elif len(negative) >= 3 and len(positive) == 0:
        return "No", negative[:3]
    elif len(positive) >= 2 and len(negative) >= 2:
        return "Maybe", positive[:2] + negative[:2]  # 展示冲突
    else:
        return "Maybe", candidates[:3]  # 证据不足
```

这种方法让系统"知道自己不知道"，而非强行给出答案。
