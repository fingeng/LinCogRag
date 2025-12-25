from src.embedding_store import EmbeddingStore
from src.utils import min_max_normalize
import os
import json
from collections import defaultdict
import numpy as np
import math
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
import igraph as ig
import re
import logging

# HyperLinearRAG imports
from src.hypergraph import (
    CooccurrenceHyperedgeBuilder,
    MedicalHyperedgeEnhancer,
    HypergraphStore,
    IncrementalIndexer,
    MultiLevelCache,
)

logger = logging.getLogger(__name__)
class LinearRAG:
    def __init__(self, global_config):
        self.config = global_config
        logger.info(f"Initializing LinearRAG with config: {self.config}")
        self.dataset_name = global_config.dataset_name
        self.load_embedding_store()
        self.llm_model = self.config.llm_model
        
        # 🔧 FIXED: Use hybrid BC5CDR + HF NER strategy based on config
        from src.ner import SpacyNER
        
        # 🔧 Check config for NER strategy
        use_hybrid = getattr(self.config, 'use_hf_ner', True) or getattr(self.config, 'use_enhanced_ner', True)
        
        if use_hybrid:
            print("[LinearRAG] Using Hybrid NER: BC5CDR (primary) + HF (supplement)")
        else:
            print("[LinearRAG] Using BC5CDR NER only")
        
        self.spacy_ner = SpacyNER(
            self.config.spacy_model,
            use_hybrid=use_hybrid
        )
        
        self.graph = ig.Graph(directed=False)
        
        # HyperLinearRAG: Initialize hypergraph components
        self.use_hypergraph = getattr(self.config, 'use_hypergraph', True)
        if self.use_hypergraph:
            print("[LinearRAG] Hypergraph mode enabled")
            hypergraph_dir = os.path.join(self.config.working_dir, self.dataset_name, "hypergraph")
            self.hypergraph_store = HypergraphStore(hypergraph_dir)
            self.hyperedge_builder = CooccurrenceHyperedgeBuilder(
                min_entities=getattr(self.config, 'min_entities_per_hyperedge', 2),
            )
            self.hyperedge_enhancer = MedicalHyperedgeEnhancer(
                max_boost=getattr(self.config, 'max_hyperedge_score_boost', 1.5)
            )
            
            # Initialize cache if enabled
            if getattr(self.config, 'enable_multi_level_cache', False):
                cache_dir = os.path.join(self.config.working_dir, self.dataset_name, "cache")
                self.cache_manager = MultiLevelCache(cache_dir)
            else:
                self.cache_manager = None
            
            # Initialize incremental indexer if enabled
            if getattr(self.config, 'enable_incremental_index', False):
                self.incremental_indexer = IncrementalIndexer(
                    os.path.join(self.config.working_dir, self.dataset_name)
                )
            else:
                self.incremental_indexer = None
            
            # Hyperedge embeddings (initialized in index)
            self.hyperedge_embeddings = None
            self.hyperedge_hash_ids = []

    def load_embedding_store(self):
        self.passage_embedding_store = EmbeddingStore(self.config.embedding_model, db_filename=os.path.join(self.config.working_dir,self.dataset_name, "passage_embedding.parquet"), batch_size=self.config.batch_size, namespace="passage")
        self.entity_embedding_store = EmbeddingStore(self.config.embedding_model, db_filename=os.path.join(self.config.working_dir,self.dataset_name, "entity_embedding.parquet"), batch_size=self.config.batch_size, namespace="entity")
        self.sentence_embedding_store = EmbeddingStore(self.config.embedding_model, db_filename=os.path.join(self.config.working_dir,self.dataset_name, "sentence_embedding.parquet"), batch_size=self.config.batch_size, namespace="sentence")

    def load_existing_data(self,passage_hash_ids):
        self.ner_results_path = os.path.join(self.config.working_dir,self.dataset_name, "ner_results.json")
        if os.path.exists(self.ner_results_path):
            existing_ner_reuslts = json.load(open(self.ner_results_path))
            existing_passage_hash_id_to_entities = existing_ner_reuslts["passage_hash_id_to_entities"]
            existing_sentence_to_entities = existing_ner_reuslts["sentence_to_entities"]
            existing_passage_hash_ids = set(existing_passage_hash_id_to_entities.keys())
            new_passage_hash_ids = set(passage_hash_ids) - existing_passage_hash_ids
            return existing_passage_hash_id_to_entities, existing_sentence_to_entities, new_passage_hash_ids
        else:
            return {}, {}, passage_hash_ids

    def qa(self, questions):
        retrieval_results = self.retrieve(questions)
        
        # 🔧 根据数据集类型分组问题
        dataset_groups = defaultdict(list)
        for idx, retrieval_result in enumerate(retrieval_results):
            dataset_name = retrieval_result.get("dataset", "unknown")
            dataset_groups[dataset_name].append((idx, retrieval_result))
        
        # 为每个数据集准备不同的 prompt
        all_messages = []
        message_to_result_idx = []
        
        for dataset_name, group_items in dataset_groups.items():
            # 🔧 根据数据集类型选择 system prompt
            if dataset_name in ["pubmedqa"]:
                system_prompt = """You are a medical expert. Answer the question based on the provided context.

IMPORTANT: You MUST respond with EXACTLY one of these three words: Yes, No, or Maybe
Do NOT add any punctuation, explanation, or other text.

Example responses:
- Yes
- No  
- Maybe"""
                answer_format = "Answer with ONLY: Yes, No, or Maybe"
                
            elif dataset_name in ["bioasq"]:
                system_prompt = """You are a medical expert. Answer the yes/no question based on the provided context.

IMPORTANT: You MUST respond with EXACTLY one of these two words: Yes or No
Do NOT add any punctuation, explanation, or other text.

Example responses:
- Yes
- No"""
                answer_format = "Answer with ONLY: Yes or No"
                
            else:  # mmlu, medqa, medmcqa
                system_prompt = """You are a medical expert. Answer the multiple-choice question based on the provided context.

CRITICAL INSTRUCTIONS:
1. You MUST respond with ONLY ONE LETTER: A, B, C, or D
2. Do NOT write "Answer: A" - just write "A"
3. Do NOT add any explanation, reasoning, or other text
4. Do NOT write "Thought:" or anything else

CORRECT examples:
- A
- B
- C
- D

INCORRECT examples (DO NOT DO THIS):
- Answer: A
- Thought: ... Answer: A
- The answer is A
- A.
- Option A"""
                answer_format = "YOUR RESPONSE MUST BE EXACTLY ONE LETTER: A, B, C, or D (nothing else)"
            
            # 为该数据集的每个问题构建 messages
            for idx, retrieval_result in group_items:
                question_text = retrieval_result["question"]
                sorted_passage = retrieval_result["sorted_passage"]
                
                # 🔧 构建 context
                prompt_user = "Context:\n"
                for passage in sorted_passage:
                    prompt_user += f"{passage}\n\n"
                
                # 🔧 添加完整的问题（包含 options）
                prompt_user += f"Question with Options:\n{question_text}\n\n"
                prompt_user += f"{answer_format}\n"
                
                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt_user}
                ]
                all_messages.append(messages)
                message_to_result_idx.append(idx)
        
        # 并行调用 LLM (限制并发数避免资源耗尽)
        llm_workers = min(2, self.config.max_workers)  # ✅ 最多2个LLM并发
        with ThreadPoolExecutor(max_workers=llm_workers) as executor:
            all_qa_results = list(tqdm(
                executor.map(self.llm_model.infer, all_messages),
                total=len(all_messages),
                desc="QA Reading (Parallel)"
            ))

        # 🔧 根据数据集类型解析答案
        for qa_result, result_idx in zip(all_qa_results, message_to_result_idx):
            retrieval_result = retrieval_results[result_idx]
            dataset_name = retrieval_result.get("dataset", "unknown")
            
            try:
                if dataset_name in ["pubmedqa"]:
                    # PubMedQA: 提取 Yes/No/Maybe
                    pred_ans = qa_result.strip()
                    match = re.search(r'\b(yes|no|maybe)\b', pred_ans, re.IGNORECASE)
                    if match:
                        pred_ans = match.group(1).capitalize()
                    else:
                        pred_ans = "INVALID"
                        
                elif dataset_name in ["bioasq"]:
                    # BioASQ: 提取 Yes/No
                    pred_ans = qa_result.strip()
                    match = re.search(r'\b(yes|no)\b', pred_ans, re.IGNORECASE)
                    if match:
                        pred_ans = match.group(1).capitalize()
                    else:
                        pred_ans = "INVALID"
                        
                else:  # mmlu, medqa, medmcqa
                    # 🔧 四选一：只提取第一个出现的 A/B/C/D
                    pred_ans = qa_result.strip().upper()
                    
                    # 直接检查是否是单个字母
                    if pred_ans in ['A', 'B', 'C', 'D']:
                        pass  # 已经是正确格式
                    else:
                        # ✅ 使用边界匹配，避免匹配到 "answer" 中的 'A'
                        match = re.search(r'\b([ABCD])\b', pred_ans)
                        if match:
                            pred_ans = match.group(1)
                        else:
                            # 如果没有边界匹配，尝试任意字母匹配
                            match = re.search(r'[ABCD]', pred_ans)
                            if match:
                                pred_ans = match.group(0)
                            else:
                                # 如果完全没有找到，记录原始回答
                                print(f"⚠️  {dataset_name}: Invalid answer format: {qa_result[:100]}")
                                pred_ans = "INVALID"
                    
            except Exception as e:
                print(f"⚠️  Error parsing answer for {dataset_name}: {e}")
                pred_ans = "INVALID"
            
            retrieval_result["pred_answer"] = pred_ans
            # 🔧 REMOVED: raw_answer field (no longer needed)
    
        return retrieval_results
        
        
        
    def retrieve(self, questions):
        retrieval_results = []
        for question_info in tqdm(questions, desc="Retrieving"):
            question = question_info["question"]
            
            question_embedding = self.config.embedding_model.encode(
                question,
                normalize_embeddings=True,
                show_progress_bar=False,
                batch_size=self.config.batch_size
            )
            
            seed_entity_indices, seed_entities, seed_entity_hash_ids, seed_entity_scores = self.get_seed_entities(question)
            
            has_entities = len(seed_entities) > 0
            hyperedge_context = ""
            
            # HyperLinearRAG: Use hybrid retrieval if hypergraph is enabled
            if self.use_hypergraph and self.hyperedge_embeddings is not None:
                seed_entity_data = (seed_entity_indices, seed_entities, seed_entity_hash_ids, seed_entity_scores)
                sorted_passage_hash_ids, sorted_passage_scores, hyperedge_context = self.hybrid_retrieve(
                    question, question_embedding, seed_entity_data
                )
                
                final_passage_hash_ids = sorted_passage_hash_ids[:self.config.retrieval_top_k]
                final_passage_scores = sorted_passage_scores[:self.config.retrieval_top_k]
                final_passages = [self.passage_embedding_store.hash_id_to_text[pid] for pid in final_passage_hash_ids]
                
            elif has_entities:
                sorted_passage_hash_ids, sorted_passage_scores = self.graph_search_with_seed_entities(
                    question_embedding, seed_entity_indices, seed_entities, seed_entity_hash_ids, seed_entity_scores
                )
                
                final_passage_hash_ids = sorted_passage_hash_ids[:self.config.retrieval_top_k]
                final_passage_scores = sorted_passage_scores[:self.config.retrieval_top_k]
                final_passages = [self.passage_embedding_store.hash_id_to_text[pid] for pid in final_passage_hash_ids]
            else:
                sorted_passage_indices, sorted_passage_scores = self.dense_passage_retrieval(question_embedding)
                final_passage_indices = sorted_passage_indices[:self.config.retrieval_top_k]
                final_passage_scores = sorted_passage_scores[:self.config.retrieval_top_k]
                final_passages = [self.passage_embedding_store.texts[idx] for idx in final_passage_indices]
            
            # Prepend hyperedge context if available
            if hyperedge_context and final_passages:
                final_passages = [hyperedge_context + "\n\n" + final_passages[0]] + final_passages[1:]
            
            result = {
                "question": question,
                "sorted_passage": final_passages,
                "sorted_passage_scores": final_passage_scores,
                "answer": question_info["answer"],
                "dataset": question_info.get("dataset", "unknown"),
                "has_entities": has_entities,
                "has_hyperedge_context": bool(hyperedge_context),
            }
            retrieval_results.append(result)
        return retrieval_results

    def expand_medical_query(self, question):
        """Expand query with medical synonyms and related terms"""
        # 🔧 Comprehensive medical term mappings
        medical_expansions = {
            # Common abbreviations
            'mi': 'myocardial infarction heart attack',
            'copd': 'chronic obstructive pulmonary disease',
            'htn': 'hypertension high blood pressure',
            'dm': 'diabetes mellitus',
            'chf': 'congestive heart failure',
            'cad': 'coronary artery disease',
            'uti': 'urinary tract infection',
            'copd': 'chronic obstructive pulmonary disease emphysema',
            'tb': 'tuberculosis',
            'hiv': 'human immunodeficiency virus aids',
            
            # Disease expansions
            'pneumonia': 'lung infection respiratory inflammation',
            'carcinoma': 'cancer tumor malignancy neoplasm',
            'adenocarcinoma': 'glandular cancer tumor',
            'lymphoma': 'lymph node cancer',
            'leukemia': 'blood cancer',
            'stroke': 'cerebrovascular accident cva brain infarction',
            'sepsis': 'blood infection septicemia',
            'asthma': 'airway obstruction bronchospasm',
            
            # Symptom expansions
            'dyspnea': 'shortness of breath breathing difficulty',
            'tachycardia': 'rapid heart rate fast heartbeat',
            'bradycardia': 'slow heart rate',
            'hypotension': 'low blood pressure',
            'hypertension': 'high blood pressure elevated bp',
            'fever': 'pyrexia elevated temperature',
            'pain': 'discomfort ache',
            'nausea': 'sick feeling vomiting',
            
            # Treatment expansions
            'antibiotic': 'antimicrobial antibacterial medication',
            'chemotherapy': 'cancer treatment cytotoxic therapy',
            'radiation': 'radiotherapy irradiation treatment',
            'surgery': 'operation surgical procedure',
        }
        
        expanded = question
        question_lower = question.lower()
        
        # Add expansions for found terms
        for term, expansion in medical_expansions.items():
            # Match whole words only
            import re
            if re.search(r'\b' + re.escape(term) + r'\b', question_lower):
                expanded += f" {expansion}"
        
        return expanded

    def clinical_rerank(self, question, passage_hash_ids, passage_scores):
        """Enhanced clinical relevance re-ranking"""
        import re
        
        # Extract clinical keywords from question
        clinical_keywords = set()
        question_lower = question.lower()
        
        # Enhanced medical patterns
        patterns = [
            r'\b\d+[-/]year[-]old\b',  # Age
            r'\b(?:mg|mcg|mL|mmHg|mm\s*Hg|g/dL)\b',  # Units
            r'\b(?:presents?|complain|history|symptom|sign)\b',  # Clinical terms
            r'\b(?:treatment|therapy|medication|drug|prescription)\b',  # Treatment
            r'\b(?:diagnosed?|diagnosis|prognosis|differential)\b',  # Diagnosis
            r'\b(?:patient|clinical|medical|case)\b',  # General medical
            r'\b(?:elevated|decreased|increased|reduced)\b',  # Lab changes
            r'\b(?:acute|chronic|recurrent|persistent)\b',  # Timing
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, question_lower, re.IGNORECASE)
            clinical_keywords.update([m.lower() for m in matches])
        
        # Calculate clinical relevance scores
        new_scores = []
        for pid, base_score in zip(passage_hash_ids, passage_scores):
            passage_text = self.passage_embedding_store.hash_id_to_text[pid].lower()
            
            # 1. Keyword overlap bonus (max +60%)
            keyword_overlap = sum(1 for kw in clinical_keywords if kw in passage_text)
            keyword_bonus = min(keyword_overlap * 0.12, 0.6)
            
            # 2. Strong recency penalty for old papers
            year_match = re.search(r'19[67]\d', passage_text)
            recency_penalty = -0.3 if year_match else 0
            
            # 3. Enhanced clinical context bonus (max +35%)
            clinical_terms = [
                'patient', 'treatment', 'therapy', 'clinical', 'diagnosis', 
                'symptoms', 'disease', 'disorder', 'condition', 'management'
            ]
            clinical_bonus = min(sum(0.05 for term in clinical_terms if term in passage_text), 0.35)
            
            # 4. Penalize pure research/methodology papers
            research_terms = ['experiment', 'methodology', 'statistical', 'protocol', 'study design']
            research_penalty = -0.2 if sum(1 for term in research_terms if term in passage_text) >= 2 else 0
            
            # 5. Boost case reports and clinical presentations
            clinical_indicators = ['case report', 'clinical presentation', 'patient presented', 'physical examination']
            case_bonus = 0.15 if any(ind in passage_text for ind in clinical_indicators) else 0
            
            # Calculate final adjusted score
            adjusted_score = base_score * (1 + keyword_bonus + recency_penalty + clinical_bonus + research_penalty + case_bonus)
            new_scores.append(adjusted_score)
        
        # Re-sort by adjusted scores
        sorted_indices = np.argsort(new_scores)[::-1]
        reranked_ids = [passage_hash_ids[i] for i in sorted_indices]
        reranked_scores = [new_scores[i] for i in sorted_indices]
        
        return reranked_ids, reranked_scores

    def graph_search_with_seed_entities(self, question_embedding, seed_entity_indices, seed_entities, seed_entity_hash_ids, seed_entity_scores):
        # ✅ 优化: 先用DPR筛选候选passage集合
        candidate_passage_hash_ids = None
        if hasattr(self.config, 'use_candidate_filtering') and self.config.use_candidate_filtering:
            dpr_indices, _ = self.dense_passage_retrieval(question_embedding)
            candidate_size = min(self.config.candidate_pool_size, len(dpr_indices))
            candidate_indices = dpr_indices[:candidate_size]
            candidate_passage_hash_ids = {
                self.passage_embedding_store.hash_ids[idx] 
                for idx in candidate_indices
            }
        
        # 计算实体权重
        entity_weights, actived_entities = self.calculate_entity_scores(
            question_embedding,
            seed_entity_indices,
            seed_entities,
            seed_entity_hash_ids,
            seed_entity_scores
        )
        
        # 计算passage权重 (只在候选集上)
        passage_weights = self.calculate_passage_scores(
            question_embedding, 
            actived_entities, 
            candidate_passage_hash_ids  # ✅ 传入候选集
        )
        
        node_weights = entity_weights + passage_weights
        ppr_sorted_passage_indices, ppr_sorted_passage_scores = self.run_ppr(node_weights)
        return ppr_sorted_passage_indices, ppr_sorted_passage_scores

    def run_ppr(self, node_weights):        
        reset_prob = np.where(np.isnan(node_weights) | (node_weights < 0), 0, node_weights)
        pagerank_scores = self.graph.personalized_pagerank(
            vertices=range(len(self.node_name_to_vertex_idx)),
            damping=self.config.damping,
            directed=False,
            weights='weight',
            reset=reset_prob,
            implementation='prpack'
        )
        
        doc_scores = np.array([pagerank_scores[idx] for idx in self.passage_node_indices])
        sorted_indices_in_doc_scores = np.argsort(doc_scores)[::-1]
        sorted_passage_scores = doc_scores[sorted_indices_in_doc_scores]
        
        sorted_passage_hash_ids = [
            self.vertex_idx_to_node_name[self.passage_node_indices[i]] 
            for i in sorted_indices_in_doc_scores
        ]
        
        return sorted_passage_hash_ids, sorted_passage_scores.tolist()

    def calculate_entity_scores(self,question_embedding,seed_entity_indices,seed_entities,seed_entity_hash_ids,seed_entity_scores):
        actived_entities = {}
        entity_weights = np.zeros(len(self.graph.vs["name"]))
        for seed_entity_idx,seed_entity,seed_entity_hash_id,seed_entity_score in zip(seed_entity_indices,seed_entities,seed_entity_hash_ids,seed_entity_scores):
            actived_entities[seed_entity_hash_id] = (seed_entity_idx, seed_entity_score, 1)
            seed_entity_node_idx = self.node_name_to_vertex_idx[seed_entity_hash_id]
            entity_weights[seed_entity_node_idx] = seed_entity_score    
        used_sentence_hash_ids = set()
        current_entities = actived_entities.copy()
        iteration = 1
        while len(current_entities) > 0 and iteration < self.config.max_iterations:
            new_entities = {}
            for entity_hash_id, (entity_id, entity_score, tier) in current_entities.items():
                if entity_score < self.config.iteration_threshold:
                    continue
                
                if entity_hash_id not in self.entity_hash_id_to_sentence_hash_ids:
                    continue
                
                sentence_hash_ids = [sid for sid in list(self.entity_hash_id_to_sentence_hash_ids[entity_hash_id]) if sid not in used_sentence_hash_ids]
                
                if not sentence_hash_ids:
                    continue
                
                valid_sentence_hash_ids = []
                for sid in sentence_hash_ids:
                    if sid in self.sentence_embedding_store.hash_id_to_idx:
                        valid_sentence_hash_ids.append(sid)
                
                if not valid_sentence_hash_ids:
                    continue
                
                sentence_indices = [self.sentence_embedding_store.hash_id_to_idx[sid] for sid in valid_sentence_hash_ids]
                sentence_embeddings = self.sentence_embeddings[sentence_indices]
                question_emb = question_embedding.reshape(-1, 1) if len(question_embedding.shape) == 1 else question_embedding
                sentence_similarities = np.dot(sentence_embeddings, question_emb).flatten()
                top_sentence_indices = np.argsort(sentence_similarities)[::-1][:self.config.top_k_sentence]
                
                for top_sentence_index in top_sentence_indices:
                    top_sentence_hash_id = valid_sentence_hash_ids[top_sentence_index]
                    top_sentence_score = sentence_similarities[top_sentence_index]
                    
                    # ✅ 优化: 过滤低相关句子
                    if top_sentence_score < 0.25:
                        continue
                    
                    used_sentence_hash_ids.add(top_sentence_hash_id)
                    
                    if top_sentence_hash_id not in self.sentence_hash_id_to_entity_hash_ids:
                        continue
                    
                    entity_hash_ids_in_sentence = self.sentence_hash_id_to_entity_hash_ids[top_sentence_hash_id]
                    
                    for next_entity_hash_id in entity_hash_ids_in_sentence:
                        next_entity_score = entity_score * top_sentence_score
                        
                        # ✅ 优化: 降低远距离实体权重
                        if tier > 1:
                            next_entity_score *= 0.7
                        
                        if next_entity_score < self.config.iteration_threshold:
                            continue
                        
                        if next_entity_hash_id not in self.node_name_to_vertex_idx:
                            continue
                        
                        next_enitity_node_idx = self.node_name_to_vertex_idx[next_entity_hash_id]
                        entity_weights[next_enitity_node_idx] += next_entity_score
                        new_entities[next_entity_hash_id] = (next_enitity_node_idx, next_entity_score, iteration+1)
            
            actived_entities.update(new_entities)
            current_entities = new_entities.copy()
            iteration += 1
        return entity_weights, actived_entities

    def calculate_passage_scores(self, question_embedding, actived_entities, candidate_passage_hash_ids=None):
        """计算passage权重，支持候选集过滤"""
        passage_weights = np.zeros(len(self.graph.vs["name"]))
        dpr_passage_indices, dpr_passage_scores = self.dense_passage_retrieval(question_embedding)
        dpr_passage_scores = min_max_normalize(dpr_passage_scores)
        
        # ✅ 如果有候选集，只处理候选passage
        for i, dpr_passage_index in enumerate(dpr_passage_indices):
            passage_hash_id = self.passage_embedding_store.hash_ids[dpr_passage_index]
            
            # ✅ 候选集过滤：跳过不在候选集中的passage
            if candidate_passage_hash_ids is not None and passage_hash_id not in candidate_passage_hash_ids:
                continue
            
            total_entity_bonus = 0
            dpr_passage_score = dpr_passage_scores[i]
            passage_text_lower = self.passage_embedding_store.hash_id_to_text[passage_hash_id].lower()
            
            for entity_hash_id, (entity_id, entity_score, tier) in actived_entities.items():
                entity_lower = self.entity_embedding_store.hash_id_to_text[entity_hash_id].lower()
                entity_occurrences = passage_text_lower.count(entity_lower)
                if entity_occurrences > 0:
                    denom = tier if tier >= 1 else 1
                    entity_bonus = entity_score * math.log(1 + entity_occurrences) / denom
                    total_entity_bonus += entity_bonus
            
            passage_score = self.config.passage_ratio * dpr_passage_score + math.log(1 + total_entity_bonus)
            passage_node_idx = self.node_name_to_vertex_idx[passage_hash_id]
            passage_weights[passage_node_idx] = passage_score * self.config.passage_node_weight
        
        return passage_weights

    def dense_passage_retrieval(self, question_embedding):
        question_emb = question_embedding.reshape(1, -1)
        question_passage_similarities = np.dot(self.passage_embeddings, question_emb.T).flatten()
        sorted_passage_indices = np.argsort(question_passage_similarities)[::-1]
        sorted_passage_scores = question_passage_similarities[sorted_passage_indices].tolist()
        return sorted_passage_indices, sorted_passage_scores
    
    def get_seed_entities(self, question):
        """Get seed entities from question with fallback"""
        question_entities = list(self.spacy_ner.question_ner(question))
        
        if not hasattr(self, 'entity_embeddings') or self.entity_embeddings is None:
            return [], [], [], []
        
        if len(self.entity_embeddings) == 0:
            return [], [], [], []
        
        if len(question_entities) == 0:
            if not hasattr(self, '_no_entity_count'):
                self._no_entity_count = 0
            self._no_entity_count += 1
            return [], [], [], []
        
        try:
            question_entity_embeddings = self.config.embedding_model.encode(
                question_entities,
                normalize_embeddings=True,
                show_progress_bar=False,
                batch_size=self.config.batch_size
            )
            
            if question_entity_embeddings.ndim == 1:
                question_entity_embeddings = question_entity_embeddings.reshape(1, -1)
                
            similarities = np.dot(self.entity_embeddings, question_entity_embeddings.T)
            
        except ValueError as e:
            print(f"⚠️  Error computing similarities: {e}")
            return [], [], [], []
        
        seed_entity_indices = []
        seed_entity_texts = []
        seed_entity_hash_ids = []
        seed_entity_scores = []
        
        for query_entity_idx in range(len(question_entities)):
            entity_scores = similarities[:, query_entity_idx]
            best_entity_idx = np.argmax(entity_scores)
            best_entity_score = entity_scores[best_entity_idx]
            best_entity_hash_id = self.entity_hash_ids[best_entity_idx]
            best_entity_text = self.entity_embedding_store.hash_id_to_text[best_entity_hash_id]
            seed_entity_indices.append(best_entity_idx)
            seed_entity_texts.append(best_entity_text)
            seed_entity_hash_ids.append(best_entity_hash_id)
            seed_entity_scores.append(best_entity_score)
        
        return seed_entity_indices, seed_entity_texts, seed_entity_hash_ids, seed_entity_scores

    def index(self, passages):
        import time
        
        print(f"\n{'='*60}")
        print(f"[LinearRAG.index] Starting indexing process")
        print(f"[LinearRAG.index] Number of passages: {len(passages)}")
        print(f"{'='*60}\n")
        
        self.node_to_node_stats = defaultdict(dict)
        self.entity_to_sentence_stats = defaultdict(dict)
        
        # Step 1-8: Insert passages, NER, build graph...
        print("[LinearRAG.index] Step 1/8: Inserting passages...")
        step_start = time.time()
        self.passage_embedding_store.insert_text(passages)
        print(f"[LinearRAG.index] ✅ Step 1 completed in {time.time()-step_start:.2f}s\n")
        
        # Load hash_id mapping
        print("[LinearRAG.index] Step 2/8: Loading mappings...")
        hash_id_to_passage = self.passage_embedding_store.get_hash_id_to_text()
        current_passage_texts = set(passages)
        current_hash_ids = {hid for hid, text in hash_id_to_passage.items() if text in current_passage_texts}
        print(f"[LinearRAG.index] ✅ Step 2 completed\n")
        
        # Load existing NER data
        print("[LinearRAG.index] Step 3/8: Loading existing NER...")
        existing_passage_hash_id_to_entities, existing_sentence_to_entities, new_passage_hash_ids = self.load_existing_data(current_hash_ids)
        print(f"[LinearRAG.index] ✅ Step 3 completed - {len(new_passage_hash_ids)} new passages\n")
        
        # Process new passages with NER
        if len(new_passage_hash_ids) > 0:
            print(f"[LinearRAG.index] Step 4/8: Running NER on {len(new_passage_hash_ids)} passages...")
            new_hash_id_to_passage = {k: hash_id_to_passage[k] for k in new_passage_hash_ids}
            new_passage_hash_id_to_entities, new_sentence_to_entities = self.spacy_ner.batch_ner(new_hash_id_to_passage, self.config.max_workers)
            self.merge_ner_results(existing_passage_hash_id_to_entities, existing_sentence_to_entities, new_passage_hash_id_to_entities, new_sentence_to_entities)
            print(f"[LinearRAG.index] ✅ Step 4 completed\n")
        else:
            print("[LinearRAG.index] Step 4/8: Skipped (no new passages)\n")
        
        # Save NER results
        print("[LinearRAG.index] Step 5/8: Saving NER...")
        self.save_ner_results(existing_passage_hash_id_to_entities, existing_sentence_to_entities)
        print(f"[LinearRAG.index] ✅ Step 5 completed\n")
        
        # Extract nodes and edges
        print("[LinearRAG.index] Step 6/9: Extracting nodes...")
        entity_nodes, sentence_nodes, passage_hash_id_to_entities, self.entity_to_sentence, self.sentence_to_entity = self.extract_nodes_and_edges(existing_passage_hash_id_to_entities, existing_sentence_to_entities)
        print(f"[LinearRAG.index] Found {len(entity_nodes)} entities, {len(sentence_nodes)} sentences\n")
        
        # HyperLinearRAG: Build hypergraph from sentence-entity co-occurrence
        if self.use_hypergraph:
            print("[LinearRAG.index] Step 6.5/9: Building hypergraph...")
            self._build_hypergraph(existing_sentence_to_entities, hash_id_to_passage)
            print(f"[LinearRAG.index] ✅ Hypergraph built\n")
        
        # Build embeddings
        print("[LinearRAG.index] Step 7/9: Building embeddings...")
        self.sentence_embedding_store.insert_text(list(sentence_nodes))
        self.entity_embedding_store.insert_text(list(entity_nodes))
        
        # Build mappings
        self.entity_hash_id_to_sentence_hash_ids = {}
        for entity, sentences in self.entity_to_sentence.items():
            entity_hash_id = self.entity_embedding_store.text_to_hash_id.get(entity)
            if entity_hash_id is None:
                continue
            sentence_hash_ids = []
            for s in sentences:
                s_hash_id = self.sentence_embedding_store.text_to_hash_id.get(s)
                if s_hash_id is not None:
                    sentence_hash_ids.append(s_hash_id)
            if sentence_hash_ids:
                self.entity_hash_id_to_sentence_hash_ids[entity_hash_id] = sentence_hash_ids
        
        self.sentence_hash_id_to_entity_hash_ids = {}
        for sentence, entities in self.sentence_to_entity.items():
            sentence_hash_id = self.sentence_embedding_store.text_to_hash_id.get(sentence)
            if sentence_hash_id is None:
                continue
            entity_hash_ids = []
            for e in entities:
                e_hash_id = self.entity_embedding_store.text_to_hash_id.get(e)
                if e_hash_id is not None:
                    entity_hash_ids.append(e_hash_id)
            if entity_hash_ids:
                self.sentence_hash_id_to_entity_hash_ids[sentence_hash_id] = entity_hash_ids
        
        # Filter passage entities
        safe_passage_hash_id_to_entities = {}
        for passage_hash_id, entities in passage_hash_id_to_entities.items():
            safe_entities = set()
            for entity in entities:
                if entity in self.entity_embedding_store.text_to_hash_id:
                    safe_entities.add(entity)
            if safe_entities:
                safe_passage_hash_id_to_entities[passage_hash_id] = safe_entities
        
        self.add_entity_to_passage_edges(safe_passage_hash_id_to_entities)
        self.add_adjacent_passage_edges()
        print(f"[LinearRAG.index] ✅ Step 7 completed\n")
        
        # Build graph
        print("[LinearRAG.index] Step 8/9: Building graph...")
        self.augment_graph()
        output_graphml_path = os.path.join(self.config.working_dir, self.dataset_name, "LinearRAG.graphml")
        os.makedirs(os.path.dirname(output_graphml_path), exist_ok=True)
        self.graph.write_graphml(output_graphml_path)
        print(f"[LinearRAG.index] ✅ Step 8 completed\n")
        
        # Load embeddings
        print("[LinearRAG.index] Loading embeddings into memory...")
        if hasattr(self.passage_embedding_store, 'get_all_embeddings'):
            self.passage_embeddings = np.array(self.passage_embedding_store.get_all_embeddings())
        else:
            self.passage_embeddings = np.array(self.passage_embedding_store.embeddings)
        
        if hasattr(self.entity_embedding_store, 'get_all_embeddings'):
            self.entity_embeddings = np.array(self.entity_embedding_store.get_all_embeddings())
        else:
            self.entity_embeddings = np.array(self.entity_embedding_store.embeddings)
        
        if hasattr(self.sentence_embedding_store, 'get_all_embeddings'):
            self.sentence_embeddings = np.array(self.sentence_embedding_store.get_all_embeddings())
        else:
            self.sentence_embeddings = np.array(self.sentence_embedding_store.embeddings)
        
        self.entity_hash_ids = self.entity_embedding_store.hash_ids
        self.vertex_idx_to_node_name = {v.index: v["name"] for v in self.graph.vs}
        
        print(f"[LinearRAG.index]   Passage embeddings: {self.passage_embeddings.shape}")
        print(f"[LinearRAG.index]   Entity embeddings: {self.entity_embeddings.shape}")
        print(f"[LinearRAG.index]   Sentence embeddings: {self.sentence_embeddings.shape}")
        
        # HyperLinearRAG: Load hyperedge embeddings and merge graph
        if self.use_hypergraph:
            print("[LinearRAG.index] Step 9/9: Loading hypergraph embeddings...")
            self._load_hyperedge_embeddings()
            print(f"[LinearRAG.index]   Hyperedge embeddings: {self.hyperedge_embeddings.shape if self.hyperedge_embeddings is not None else 'None'}")
        
        print(f"{'='*60}")
        print(f"[LinearRAG.index] ✅ Indexing completed!")
        print(f"{'='*60}\n")

    def add_adjacent_passage_edges(self):
        passage_id_to_text = self.passage_embedding_store.get_hash_id_to_text()
        index_pattern = re.compile(r'^(\d+):')
        indexed_items = [
            (int(match.group(1)), node_key)
            for node_key, text in passage_id_to_text.items()
            if (match := index_pattern.match(text.strip()))
        ]
        indexed_items.sort(key=lambda x: x[0])
        for i in range(len(indexed_items) - 1):
            current_node = indexed_items[i][1]
            next_node = indexed_items[i + 1][1]
            self.node_to_node_stats[current_node][next_node] = 1.0

    def augment_graph(self):
        self.add_nodes()
        self.add_edges()

    def add_nodes(self):
        existing_nodes = {v["name"]: v for v in self.graph.vs if "name" in v.attributes()}
        entity_hash_id_to_text = self.entity_embedding_store.get_hash_id_to_text()
        passage_hash_id_to_text = self.passage_embedding_store.get_hash_id_to_text()
        all_hash_id_to_text = {**entity_hash_id_to_text, **passage_hash_id_to_text}
        passage_hash_ids = set(passage_hash_id_to_text.keys())
        
        for hash_id, text in all_hash_id_to_text.items():
            if hash_id not in existing_nodes:
                self.graph.add_vertex(name=hash_id, content=text)
        
        self.node_name_to_vertex_idx = {v["name"]: v.index for v in self.graph.vs if "name" in v.attributes()}
        self.passage_node_indices = [
            self.node_name_to_vertex_idx[passage_id]
            for passage_id in passage_hash_ids
            if passage_id in self.node_name_to_vertex_idx
        ]

    def add_edges(self):
        edges = []
        weights = []
        for node_hash_id, node_to_node_stats in self.node_to_node_stats.items():
            for neighbor_hash_id, weight in node_to_node_stats.items():
                if node_hash_id == neighbor_hash_id:
                    continue
                edges.append((node_hash_id, neighbor_hash_id))
                weights.append(weight)
        self.graph.add_edges(edges)
        self.graph.es['weight'] = weights

    def add_entity_to_passage_edges(self, passage_hash_id_to_entities):
        passage_to_entity_count = {}
        passage_to_all_score = defaultdict(int)
        
        for passage_hash_id, entities in passage_hash_id_to_entities.items():
            passage = self.passage_embedding_store.hash_id_to_text[passage_hash_id]
            for entity in entities:
                if entity not in self.entity_embedding_store.text_to_hash_id:
                    continue
                entity_hash_id = self.entity_embedding_store.text_to_hash_id[entity]
                count = passage.count(entity)
                if count > 0:
                    passage_to_entity_count[(passage_hash_id, entity_hash_id)] = count
                    passage_to_all_score[passage_hash_id] += count
        
        for (passage_hash_id, entity_hash_id), count in passage_to_entity_count.items():
            total_count = passage_to_all_score[passage_hash_id]
            if total_count > 0:
                score = count / total_count
                self.node_to_node_stats[passage_hash_id][entity_hash_id] = score

    def extract_nodes_and_edges(self, existing_passage_hash_id_to_entities, existing_sentence_to_entities):
        entity_nodes = set()
        sentence_nodes = set()
        passage_hash_id_to_entities = defaultdict(set)
        entity_to_sentence = defaultdict(set)
        sentence_to_entity = defaultdict(set)
        
        for passage_hash_id, entities in existing_passage_hash_id_to_entities.items():
            for entity in entities:
                entity_nodes.add(entity)
                passage_hash_id_to_entities[passage_hash_id].add(entity)
        
        for sentence, entities in existing_sentence_to_entities.items():
            sentence_nodes.add(sentence)
            for entity in entities:
                entity_to_sentence[entity].add(sentence)
                sentence_to_entity[sentence].add(entity)
        
        return entity_nodes, sentence_nodes, passage_hash_id_to_entities, entity_to_sentence, sentence_to_entity

    def merge_ner_results(self, existing_passage_hash_id_to_entities, existing_sentence_to_entities, new_passage_hash_id_to_entities, new_sentence_to_entities):
        existing_passage_hash_id_to_entities.update(new_passage_hash_id_to_entities)
        existing_sentence_to_entities.update(new_sentence_to_entities)
        return existing_passage_hash_id_to_entities, existing_sentence_to_entities

    def save_ner_results(self, existing_passage_hash_id_to_entities, existing_sentence_to_entities):
        serializable_passage_entities = {
            k: list(v) if isinstance(v, set) else v
            for k, v in existing_passage_hash_id_to_entities.items()
        }
        serializable_sentence_entities = {
            k: list(v) if isinstance(v, set) else v
            for k, v in existing_sentence_to_entities.items()
        }
        with open(self.ner_results_path, "w") as f:
            json.dump({
                "passage_hash_id_to_entities": serializable_passage_entities,
                "sentence_to_entities": serializable_sentence_entities
            }, f)
    
    # ==================== HyperLinearRAG Methods ====================
    
    def _build_hypergraph(self, sentence_to_entities, hash_id_to_passage):
        """
        Build hypergraph from sentence-entity co-occurrence.
        Zero LLM token cost - fully local processing.
        """
        if not self.use_hypergraph:
            return
        
        # Check if hypergraph already exists (cache)
        if self.hypergraph_store.load():
            stats = self.hypergraph_store.get_stats()
            if stats.num_hyperedges > 0:
                print(f"[HyperLinearRAG] ✅ Loaded cached hypergraph: {stats.num_hyperedges} hyperedges, "
                      f"{stats.num_entities} entities")
                # Load passage mapping if exists
                mapping_path = os.path.join(self.hypergraph_store.storage_dir, "passage_to_hyperedge.json")
                if os.path.exists(mapping_path):
                    with open(mapping_path, "r") as f:
                        self.passage_to_hyperedge_ids = json.load(f)
                    print(f"[HyperLinearRAG] ✅ Loaded passage mapping: {len(self.passage_to_hyperedge_ids)} passages")
                return
        
        print("[HyperLinearRAG] Building hypergraph from scratch...")
        
        # Convert sentence_to_entities to proper format
        sentence_entities_dict = {}
        for sent, entities in sentence_to_entities.items():
            if isinstance(entities, (list, set)):
                sentence_entities_dict[sent] = set(entities)
            else:
                sentence_entities_dict[sent] = {entities}
        
        # Build hyperedges from co-occurrence
        hyperedges = self.hyperedge_builder.build_from_ner_results(sentence_entities_dict)
        
        if not hyperedges:
            logger.warning("No hyperedges built from sentence co-occurrence")
            return
        
        # Enhance scores with medical patterns
        hyperedges = self.hyperedge_enhancer.enhance_hyperedges(hyperedges)
        
        # Add to hypergraph store
        self.hypergraph_store.add_hyperedges(hyperedges)
        
        # Build passage to hyperedge mapping (optimized with set lookup)
        print(f"[HyperLinearRAG] Building passage-hyperedge mapping for {len(hash_id_to_passage)} passages...")
        passage_to_hyperedge_ids = {}
        
        # Create sentence to hyperedge mapping for O(1) lookup
        sentence_to_he_id = {he.text: he.hash_id for he in hyperedges}
        
        from tqdm import tqdm
        for passage_hash_id, passage_text in tqdm(hash_id_to_passage.items(), desc="Mapping passages"):
            matching_he_ids = []
            for sent, he_id in sentence_to_he_id.items():
                if sent in passage_text:
                    matching_he_ids.append(he_id)
            if matching_he_ids:
                passage_to_hyperedge_ids[passage_hash_id] = matching_he_ids
        
        # Save hypergraph
        self.hypergraph_store.save()
        
        # Save passage mapping
        mapping_path = os.path.join(self.hypergraph_store.storage_dir, "passage_to_hyperedge.json")
        with open(mapping_path, "w") as f:
            json.dump(passage_to_hyperedge_ids, f)
        
        # Store passage mapping for later use
        self.passage_to_hyperedge_ids = passage_to_hyperedge_ids
        
        stats = self.hypergraph_store.get_stats()
        print(f"[HyperLinearRAG] Built hypergraph: {stats.num_hyperedges} hyperedges, "
              f"{stats.num_entities} entities, avg {stats.avg_entities_per_hyperedge:.2f} entities/hyperedge")
    
    def _load_hyperedge_embeddings(self):
        """Load or compute hyperedge embeddings."""
        if not self.use_hypergraph:
            return
        
        # Get all hyperedge texts
        hyperedge_ids = self.hypergraph_store.get_all_hyperedge_ids()
        if not hyperedge_ids:
            logger.warning("No hyperedges to embed")
            self.hyperedge_embeddings = None
            return
        
        hyperedge_texts = []
        self.hyperedge_hash_ids = []
        
        for he_id in hyperedge_ids:
            text = self.hypergraph_store.get_hyperedge_text(he_id)
            if text:
                hyperedge_texts.append(text)
                self.hyperedge_hash_ids.append(he_id)
        
        if not hyperedge_texts:
            self.hyperedge_embeddings = None
            return
        
        # Compute embeddings
        print(f"[HyperLinearRAG] Computing embeddings for {len(hyperedge_texts)} hyperedges...")
        self.hyperedge_embeddings = self.config.embedding_model.encode(
            hyperedge_texts,
            normalize_embeddings=True,
            show_progress_bar=True,
            batch_size=self.config.batch_size
        )
        
        # Build hash_id to index mapping
        self.hyperedge_hash_to_idx = {
            he_id: idx for idx, he_id in enumerate(self.hyperedge_hash_ids)
        }
    
    def hypergraph_retrieve(self, question_embedding, seed_entity_hash_ids=None):
        """
        Retrieve relevant hyperedges and their entities.
        
        Args:
            question_embedding: Question embedding vector
            seed_entity_hash_ids: Optional seed entities from NER
        
        Returns:
            Tuple of (hyperedge_texts, hyperedge_scores, expanded_entity_ids)
        """
        if not self.use_hypergraph or self.hyperedge_embeddings is None:
            return [], [], set()
        
        hyperedge_top_k = getattr(self.config, 'hyperedge_top_k', 30)
        hyperedge_threshold = getattr(self.config, 'hyperedge_retrieval_threshold', 0.3)
        
        # Compute similarity with all hyperedges
        q_emb = question_embedding.reshape(1, -1) if len(question_embedding.shape) == 1 else question_embedding
        hyperedge_scores = np.dot(self.hyperedge_embeddings, q_emb.T).flatten()
        
        # Apply confidence score weighting
        for idx, he_id in enumerate(self.hyperedge_hash_ids):
            conf_score = self.hypergraph_store.get_hyperedge_score(he_id)
            hyperedge_scores[idx] *= conf_score
        
        # Get top-k hyperedges above threshold
        sorted_indices = np.argsort(hyperedge_scores)[::-1]
        
        top_hyperedge_ids = []
        top_hyperedge_texts = []
        top_hyperedge_scores = []
        
        for idx in sorted_indices[:hyperedge_top_k]:
            score = hyperedge_scores[idx]
            if score < hyperedge_threshold:
                break
            
            he_id = self.hyperedge_hash_ids[idx]
            he_text = self.hypergraph_store.get_hyperedge_text(he_id)
            
            top_hyperedge_ids.append(he_id)
            top_hyperedge_texts.append(he_text)
            top_hyperedge_scores.append(float(score))
        
        # Bidirectional expansion
        expanded_entity_ids = set()
        
        # Expand from retrieved hyperedges
        for he_id in top_hyperedge_ids:
            entity_ids = self.hypergraph_store.get_entities_by_hyperedge(he_id)
            expanded_entity_ids.update(entity_ids)
        
        # Expand from seed entities (if provided)
        if seed_entity_hash_ids:
            for entity_id in seed_entity_hash_ids:
                related_he_ids = self.hypergraph_store.get_hyperedges_by_entity(entity_id)
                for he_id in related_he_ids:
                    if he_id not in top_hyperedge_ids:
                        he_text = self.hypergraph_store.get_hyperedge_text(he_id)
                        if he_text:
                            top_hyperedge_texts.append(he_text)
                            top_hyperedge_scores.append(0.5)  # Lower score for expanded
        
        return top_hyperedge_texts, top_hyperedge_scores, expanded_entity_ids
    
    def hybrid_retrieve(self, question, question_embedding, seed_entity_data):
        """
        Hybrid retrieval: LinearRAG PPR + Hypergraph retrieval.
        
        Args:
            question: Question text
            question_embedding: Question embedding
            seed_entity_data: Tuple of (indices, texts, hash_ids, scores)
        
        Returns:
            Tuple of (passage_hash_ids, passage_scores, hyperedge_context)
        """
        seed_entity_indices, seed_entities, seed_entity_hash_ids, seed_entity_scores = seed_entity_data
        
        # Original LinearRAG retrieval
        if len(seed_entities) > 0:
            passage_hash_ids, passage_scores = self.graph_search_with_seed_entities(
                question_embedding, seed_entity_indices, seed_entities, 
                seed_entity_hash_ids, seed_entity_scores
            )
        else:
            sorted_indices, sorted_scores = self.dense_passage_retrieval(question_embedding)
            passage_hash_ids = [
                self.passage_embedding_store.hash_ids[idx] 
                for idx in sorted_indices[:self.config.retrieval_top_k * 2]
            ]
            passage_scores = sorted_scores[:self.config.retrieval_top_k * 2]
        
        # Hypergraph retrieval (if enabled)
        hyperedge_context = ""
        if self.use_hypergraph and self.hyperedge_embeddings is not None:
            hyperedge_texts, hyperedge_scores, expanded_entities = self.hypergraph_retrieve(
                question_embedding, seed_entity_hash_ids
            )
            
            # Boost passages containing expanded entities
            if expanded_entities:
                passage_hash_ids, passage_scores = self._boost_passages_with_entities(
                    passage_hash_ids, passage_scores, expanded_entities
                )
            
            # Format hyperedge context for generation
            if hyperedge_texts:
                hyperedge_context = self._format_hyperedge_context(hyperedge_texts, hyperedge_scores)
        
        return passage_hash_ids, passage_scores, hyperedge_context
    
    def _boost_passages_with_entities(self, passage_hash_ids, passage_scores, entity_ids):
        """Boost passages that contain expanded entities."""
        boost_factor = getattr(self.config, 'hyperedge_entity_boost', 1.2)
        
        boosted_scores = []
        for pid, score in zip(passage_hash_ids, passage_scores):
            passage_text = self.passage_embedding_store.hash_id_to_text.get(pid, "").lower()
            
            # Check for entity matches
            entity_matches = 0
            for entity_id in entity_ids:
                entity_text = self.hypergraph_store.get_entity_text(entity_id)
                if entity_text and entity_text.lower() in passage_text:
                    entity_matches += 1
            
            # Apply boost
            if entity_matches > 0:
                score *= (1 + (boost_factor - 1) * min(entity_matches, 3) / 3)
            
            boosted_scores.append(score)
        
        # Re-sort by boosted scores
        sorted_pairs = sorted(
            zip(passage_hash_ids, boosted_scores), 
            key=lambda x: x[1], 
            reverse=True
        )
        
        return [p[0] for p in sorted_pairs], [p[1] for p in sorted_pairs]
    
    def _format_hyperedge_context(self, hyperedge_texts, scores, max_hyperedges=5):
        """Format hyperedge texts as context for generation."""
        if not hyperedge_texts:
            return ""
        
        # Take top hyperedges
        top_texts = hyperedge_texts[:max_hyperedges]
        
        context = "[Medical Knowledge Facts]\n"
        for i, text in enumerate(top_texts, 1):
            # Truncate if too long
            if len(text) > 200:
                text = text[:200] + "..."
            context += f"{i}. {text}\n"
        
        return context
    
    def unified_ppr_with_hyperedges(self, question_embedding, seed_entity_data, hyperedge_data=None):
        """
        Unified Personalized PageRank with entity, passage, and hyperedge nodes.
        
        Args:
            question_embedding: Question embedding vector
            seed_entity_data: Tuple of (indices, texts, hash_ids, scores)
            hyperedge_data: Optional tuple of (hyperedge_ids, hyperedge_scores)
        
        Returns:
            Tuple of (sorted_passage_hash_ids, sorted_passage_scores)
        """
        seed_entity_indices, seed_entities, seed_entity_hash_ids, seed_entity_scores = seed_entity_data
        
        # Calculate entity weights (original LinearRAG logic)
        entity_weights, actived_entities = self.calculate_entity_scores(
            question_embedding, seed_entity_indices, seed_entities, 
            seed_entity_hash_ids, seed_entity_scores
        )
        
        # Calculate passage weights
        passage_weights = self.calculate_passage_scores(
            question_embedding, actived_entities, None
        )
        
        # Combine weights
        node_weights = entity_weights + passage_weights
        
        # Add hyperedge weights if enabled
        if self.use_hypergraph and hyperedge_data is not None:
            hyperedge_ids, hyperedge_scores = hyperedge_data
            hyperedge_node_weight = getattr(self.config, 'hyperedge_node_weight', 1.2)
            
            for he_id, he_score in zip(hyperedge_ids, hyperedge_scores):
                if he_id in self.node_name_to_vertex_idx:
                    he_node_idx = self.node_name_to_vertex_idx[he_id]
                    node_weights[he_node_idx] = he_score * hyperedge_node_weight
        
        # Run PPR
        ppr_sorted_passage_ids, ppr_sorted_passage_scores = self.run_ppr(node_weights)
        return ppr_sorted_passage_ids, ppr_sorted_passage_scores
    
    def build_unified_graph_with_hyperedges(self):
        """
        Merge hypergraph nodes into the main graph for unified PPR.
        Call this after both index() and hypergraph building.
        """
        if not self.use_hypergraph:
            return
        
        if not hasattr(self, 'passage_to_hyperedge_ids'):
            return
        
        # Merge hypergraph into main graph
        self.graph = self.hypergraph_store.merge_with_linear_graph(
            self.graph, 
            self.passage_to_hyperedge_ids
        )
        
        # Rebuild mappings
        self.node_name_to_vertex_idx = {
            v["name"]: v.index for v in self.graph.vs if "name" in v.attributes()
        }
        
        # Update passage node indices (filter out hyperedge nodes)
        passage_hash_ids = set(self.passage_embedding_store.hash_ids)
        self.passage_node_indices = [
            self.node_name_to_vertex_idx[pid]
            for pid in passage_hash_ids
            if pid in self.node_name_to_vertex_idx
        ]
        
        logger.info(f"Unified graph: {self.graph.vcount()} nodes, {self.graph.ecount()} edges")
