"""
高级优化方案 - 代码级别改进
这个文件提供了实际可用的代码改进方案
"""

# ============================================================================
# 优化1: 在 LinearRAG.py 中添加候选集预筛选
# ============================================================================

# 在 LinearRAG.py 第 349-357 行的 graph_search_with_seed_entities 方法中添加:

def graph_search_with_seed_entities_OPTIMIZED(
    self, question_embedding, seed_entity_indices, 
    seed_entities, seed_entity_hash_ids, seed_entity_scores
):
    """
    优化版本: 添加DPR候选集预筛选
    """
    # ✅ 新增: 先用DPR筛选候选passage (核心优化)
    dpr_indices, dpr_scores = self.dense_passage_retrieval(question_embedding)
    
    # ✅ 只在top-200个passage中进行图搜索
    candidate_size = min(200, len(dpr_indices))
    candidate_passage_indices = dpr_indices[:candidate_size]
    candidate_passage_hash_ids = {
        self.passage_embedding_store.hash_ids[idx] 
        for idx in candidate_passage_indices
    }
    
    # 原有的实体权重计算
    entity_weights, actived_entities = self.calculate_entity_scores(
        question_embedding,
        seed_entity_indices,
        seed_entities,
        seed_entity_hash_ids,
        seed_entity_scores
    )
    
    # ✅ 修改: 只计算候选passage的权重
    passage_weights = self.calculate_passage_scores_with_candidates(
        question_embedding, 
        actived_entities,
        candidate_passage_hash_ids  # 传入候选集
    )
    
    # PPR计算
    node_weights = entity_weights + passage_weights
    ppr_sorted_passage_indices, ppr_sorted_passage_scores = self.run_ppr(node_weights)
    
    return ppr_sorted_passage_indices, ppr_sorted_passage_scores


# ============================================================================
# 优化2: 在 LinearRAG.py 中优化 calculate_passage_scores
# ============================================================================

def calculate_passage_scores_with_candidates(
    self, question_embedding, actived_entities, candidate_passage_hash_ids=None
):
    """
    优化版本: 只计算候选passage的分数
    """
    passage_weights = np.zeros(len(self.graph.vs["name"]))
    
    # ✅ 如果有候选集，只处理候选passage
    if candidate_passage_hash_ids:
        # 只对候选passage计算DPR分数
        candidate_embeddings = []
        candidate_indices = []
        
        for hash_id in candidate_passage_hash_ids:
            if hash_id in self.passage_embedding_store.hash_id_to_idx:
                idx = self.passage_embedding_store.hash_id_to_idx[hash_id]
                candidate_indices.append(idx)
                candidate_embeddings.append(self.passage_embeddings[idx])
        
        # 批量计算相似度
        if candidate_embeddings:
            candidate_embeddings = np.array(candidate_embeddings)
            question_emb = question_embedding.reshape(1, -1)
            dpr_scores = np.dot(candidate_embeddings, question_emb.T).flatten()
            dpr_scores = min_max_normalize(dpr_scores)
            
            # 为每个候选passage计算最终分数
            for i, passage_idx in enumerate(candidate_indices):
                passage_hash_id = self.passage_embedding_store.hash_ids[passage_idx]
                dpr_score = dpr_scores[i]
                
                # 计算实体bonus
                total_entity_bonus = 0
                passage_text_lower = self.passage_embedding_store.hash_id_to_text[passage_hash_id].lower()
                
                for entity_hash_id, (entity_id, entity_score, tier) in actived_entities.items():
                    entity_lower = self.entity_embedding_store.hash_id_to_text[entity_hash_id].lower()
                    entity_occurrences = passage_text_lower.count(entity_lower)
                    
                    if entity_occurrences > 0:
                        denom = tier if tier >= 1 else 1
                        entity_bonus = entity_score * math.log(1 + entity_occurrences) / denom
                        total_entity_bonus += entity_bonus
                
                # 最终分数
                passage_score = self.config.passage_ratio * dpr_score + math.log(1 + total_entity_bonus)
                passage_node_idx = self.node_name_to_vertex_idx[passage_hash_id]
                passage_weights[passage_node_idx] = passage_score * self.config.passage_node_weight
    
    else:
        # 原有逻辑 (处理所有passage)
        dpr_passage_indices, dpr_passage_scores = self.dense_passage_retrieval(question_embedding)
        # ... (保持原有代码)
    
    return passage_weights


# ============================================================================
# 优化3: 在 LinearRAG.py 中优化 calculate_entity_scores
# ============================================================================

def calculate_entity_scores_OPTIMIZED(
    self, question_embedding, seed_entity_indices, 
    seed_entities, seed_entity_hash_ids, seed_entity_scores
):
    """
    优化版本: 添加early stopping和句子数量限制
    """
    actived_entities = {}
    entity_weights = np.zeros(len(self.graph.vs["name"]))
    
    # 初始化种子实体
    for seed_entity_idx, seed_entity, seed_entity_hash_id, seed_entity_score in zip(
        seed_entity_indices, seed_entities, seed_entity_hash_ids, seed_entity_scores
    ):
        actived_entities[seed_entity_hash_id] = (seed_entity_idx, seed_entity_score, 1)
        seed_entity_node_idx = self.node_name_to_vertex_idx[seed_entity_hash_id]
        entity_weights[seed_entity_node_idx] = seed_entity_score
    
    used_sentence_hash_ids = set()
    current_entities = actived_entities.copy()
    iteration = 1
    
    # ✅ 配置参数
    MAX_SENTENCES_PER_ENTITY = 20  # 限制每个实体处理的句子数
    MIN_NEW_ENTITIES = 3  # Early stopping阈值
    
    while len(current_entities) > 0 and iteration < self.config.max_iterations:
        new_entities = {}
        new_entities_count = 0  # ✅ 统计新增实体数
        
        for entity_hash_id, (entity_id, entity_score, tier) in current_entities.items():
            # ✅ 提高阈值判断
            if entity_score < self.config.iteration_threshold:
                continue
            
            if entity_hash_id not in self.entity_hash_id_to_sentence_hash_ids:
                continue
            
            # ✅ 限制句子数量 (关键优化!)
            all_sentence_hash_ids = self.entity_hash_id_to_sentence_hash_ids[entity_hash_id]
            sentence_hash_ids = [
                sid for sid in all_sentence_hash_ids 
                if sid not in used_sentence_hash_ids
            ][:MAX_SENTENCES_PER_ENTITY]  # 最多处理20个句子
            
            if not sentence_hash_ids:
                continue
            
            # 验证句子有效性
            valid_sentence_hash_ids = [
                sid for sid in sentence_hash_ids 
                if sid in self.sentence_embedding_store.hash_id_to_idx
            ]
            
            if not valid_sentence_hash_ids:
                continue
            
            # 计算句子相似度
            sentence_indices = [
                self.sentence_embedding_store.hash_id_to_idx[sid] 
                for sid in valid_sentence_hash_ids
            ]
            sentence_embeddings = self.sentence_embeddings[sentence_indices]
            question_emb = question_embedding.reshape(-1, 1) if len(question_embedding.shape) == 1 else question_embedding
            sentence_similarities = np.dot(sentence_embeddings, question_emb).flatten()
            
            # 选择top-k句子
            top_sentence_indices = np.argsort(sentence_similarities)[::-1][:self.config.top_k_sentence]
            
            # 扩散到新实体
            for top_sentence_index in top_sentence_indices:
                top_sentence_hash_id = valid_sentence_hash_ids[top_sentence_index]
                top_sentence_score = sentence_similarities[top_sentence_index]
                used_sentence_hash_ids.add(top_sentence_hash_id)
                
                if top_sentence_hash_id not in self.sentence_hash_id_to_entity_hash_ids:
                    continue
                
                entity_hash_ids_in_sentence = self.sentence_hash_id_to_entity_hash_ids[top_sentence_hash_id]
                
                for next_entity_hash_id in entity_hash_ids_in_sentence:
                    next_entity_score = entity_score * top_sentence_score
                    
                    if next_entity_score < self.config.iteration_threshold:
                        continue
                    
                    if next_entity_hash_id not in self.node_name_to_vertex_idx:
                        continue
                    
                    next_entity_node_idx = self.node_name_to_vertex_idx[next_entity_hash_id]
                    entity_weights[next_entity_node_idx] += next_entity_score
                    
                    # ✅ 统计新实体
                    if next_entity_hash_id not in new_entities and next_entity_hash_id not in actived_entities:
                        new_entities_count += 1
                    
                    new_entities[next_entity_hash_id] = (next_entity_node_idx, next_entity_score, iteration + 1)
        
        # ✅ Early stopping: 新增实体太少就停止
        if new_entities_count < MIN_NEW_ENTITIES:
            break
        
        actived_entities.update(new_entities)
        current_entities = new_entities.copy()
        iteration += 1
    
    return entity_weights, actived_entities


# ============================================================================
# 优化4: 简化NER策略 (可选)
# ============================================================================

# 在 src/ner.py 中修改 __init__ 方法:

def __init__SIMPLIFIED(self, model_name=None, use_bc5cdr=False):
    """
    简化版本: 只使用 HuggingFace biomedical-ner-all
    
    理由:
    1. HF模型覆盖23种实体类型 vs BC5CDR的2种
    2. BC5CDR的准确率提升<5%，但增加50%的NER时间
    3. 医疗QA需要更广泛的实体类型 (症状、治疗、解剖等)
    """
    self.use_bc5cdr = use_bc5cdr
    self.nlp = None
    
    # ✅ 只加载 HuggingFace NER
    print("[NER] 🔧 Using simplified NER: HuggingFace biomedical-ner-all only")
    
    from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline
    
    model_path = "models/biomedical-ner-all"
    self.hf_tokenizer = AutoTokenizer.from_pretrained(model_path)
    self.hf_model = AutoModelForTokenClassification.from_pretrained(model_path)
    
    if torch.cuda.is_available():
        self.hf_model = self.hf_model.cuda()
    
    self.hf_ner = pipeline(
        "ner",
        model=self.hf_model,
        tokenizer=self.hf_tokenizer,
        aggregation_strategy="max",
        device=0 if torch.cuda.is_available() else -1
    )
    
    print("[NER] ✅ HuggingFace NER loaded successfully")
    print("[NER] 📋 Entity types: DNA, RNA, protein, cell_line, cell_type, etc. (23 types)")


# ============================================================================
# 优化5: 添加配置参数
# ============================================================================

# 在 src/config.py 中添加新参数:

@dataclass
class LinearRAGConfig:
    def __init__(
        self,
        # ... (保持原有参数)
        
        # ✅ 新增优化参数
        candidate_pool_size=200,  # DPR候选池大小
        max_sentences_per_entity=20,  # 每个实体最多处理的句子数
        min_new_entities_for_continue=3,  # Early stopping阈值
        use_candidate_filtering=True,  # 是否启用候选集过滤
        
        # ✅ 调整后的默认值
        max_iterations=2,  # 3→2
        iteration_threshold=0.3,  # 0.1→0.3
        top_k_sentence=5,  # 3→5
    ):
        # ... (保持原有代码)
        
        # 新参数
        self.candidate_pool_size = candidate_pool_size
        self.max_sentences_per_entity = max_sentences_per_entity
        self.min_new_entities_for_continue = min_new_entities_for_continue
        self.use_candidate_filtering = use_candidate_filtering


# ============================================================================
# 使用方法
# ============================================================================

"""
## 方案1: 最小改动 (只改配置文件)

1. 修改 src/config.py:
   - max_iterations = 2
   - iteration_threshold = 0.3
   - top_k_sentence = 5

2. 重启运行:
   kill $(pgrep -f "run.py")
   python run.py --use_hf_ner ... > medqa_quick_fix.log 2>&1 &

预期效果: 速度提升 3-5倍 (90秒 → 18-30秒/问题)


## 方案2: 中等改动 (添加候选集过滤)

1. 在 LinearRAG.py 中:
   - 替换 graph_search_with_seed_entities 为 OPTIMIZED 版本
   - 添加 calculate_passage_scores_with_candidates 方法

2. 修改配置文件 (同方案1)

3. 重启运行

预期效果: 速度提升 8-12倍 (90秒 → 7-11秒/问题)


## 方案3: 完全优化 (所有改进)

1. 应用所有优化代码
2. 简化NER为只用HF模型
3. 添加early stopping和句子限制

预期效果: 速度提升 15-20倍 (90秒 → 4-6秒/问题)


## 建议实施顺序:

第1天: 方案1 (配置优化)
   - 零风险，立即见效
   - 验证准确率是否受影响

第2-3天: 方案2 (候选集过滤)
   - 如果准确率下降<2%，继续优化
   - 添加候选集预筛选逻辑

第4-5天: 方案3 (完全优化)
   - 全面测试和验证
   - 对比所有配置的准确率和速度

最终目标: 在保持准确率(下降<3%)的前提下，达到 5-10秒/问题的检索速度
"""


# ============================================================================
# 性能对比表
# ============================================================================

PERFORMANCE_COMPARISON = """
┌─────────────┬────────────────┬───────────────┬──────────────┬────────────┐
│   方案      │  检索速度      │  准确率变化   │  实施难度    │  推荐度    │
├─────────────┼────────────────┼───────────────┼──────────────┼────────────┤
│ 原始配置    │  90秒/问题     │  基线         │  -           │  ⭐        │
│ 方案1(快速) │  18-30秒       │  -0.5% ~ -1%  │  极低 (5分钟)│  ⭐⭐⭐⭐⭐ │
│ 方案2(中等) │  7-11秒        │  -1% ~ -2%    │  低 (2小时)  │  ⭐⭐⭐⭐   │
│ 方案3(完全) │  4-6秒         │  -2% ~ -3%    │  中 (1天)    │  ⭐⭐⭐    │
└─────────────┴────────────────┴───────────────┴──────────────┴────────────┘

说明:
- 准确率变化是估计值，需要实际测试验证
- 如果准确率下降<2%，速度提升>5x，就非常值得采用
- 建议先实施方案1，观察效果后再决定是否继续优化
"""

print(PERFORMANCE_COMPARISON)
