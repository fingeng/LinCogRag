"""
Advanced Retrieval Strategies for Medical QA.
Focus on improving retrieval quality and evidence utilization.
"""

import re
import numpy as np
from typing import List, Dict, Tuple, Optional, Set
from collections import defaultdict
import logging

logger = logging.getLogger(__name__)


class QueryEnhancer:
    """
    增强查询质量，提取核心医学概念并扩展。
    """
    
    # 医学术语同义词扩展
    MEDICAL_SYNONYMS = {
        # 疾病
        'heart attack': ['myocardial infarction', 'MI', 'acute coronary syndrome'],
        'stroke': ['cerebrovascular accident', 'CVA', 'brain infarction'],
        'diabetes': ['diabetes mellitus', 'DM', 'hyperglycemia'],
        'high blood pressure': ['hypertension', 'HTN'],
        'cancer': ['carcinoma', 'malignancy', 'neoplasm', 'tumor'],
        # 症状
        'chest pain': ['angina', 'thoracic pain'],
        'shortness of breath': ['dyspnea', 'breathlessness', 'respiratory distress'],
        'fever': ['pyrexia', 'febrile', 'hyperthermia'],
        # 治疗
        'surgery': ['surgical intervention', 'operation', 'procedure'],
        'medication': ['drug', 'pharmaceutical', 'treatment'],
    }
    
    # 问题类型关键词
    QUESTION_TYPE_KEYWORDS = {
        'mechanism': ['mechanism', 'how does', 'pathway', 'pathophysiology'],
        'treatment': ['treatment', 'therapy', 'manage', 'treat', 'drug'],
        'diagnosis': ['diagnose', 'diagnosis', 'test', 'confirm', 'detect'],
        'prognosis': ['prognosis', 'outcome', 'survival', 'mortality'],
        'etiology': ['cause', 'etiology', 'risk factor', 'associated with'],
    }
    
    def enhance_query(self, question: str, options: Optional[Dict[str, str]] = None) -> str:
        """
        增强查询：提取关键概念 + 同义词扩展 + 问题类型识别。
        """
        enhanced = question
        question_lower = question.lower()
        
        # 1. 同义词扩展
        for term, synonyms in self.MEDICAL_SYNONYMS.items():
            if term in question_lower:
                enhanced += " " + " ".join(synonyms)
        
        # 2. 如果有选项，将选项关键词加入查询
        if options:
            option_keywords = []
            for opt_text in options.values():
                # 提取选项中的关键医学术语（长度>4的词）
                words = re.findall(r'\b[a-zA-Z]{5,}\b', opt_text)
                option_keywords.extend(words[:3])  # 每个选项最多3个词
            if option_keywords:
                enhanced += " " + " ".join(set(option_keywords))
        
        return enhanced
    
    def extract_core_concepts(self, question: str) -> List[str]:
        """
        提取问题中的核心医学概念。
        """
        concepts = []
        question_lower = question.lower()
        
        # 使用正则提取可能的医学术语
        # 模式：首字母大写的词组（可能是专有名词）
        proper_nouns = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', question)
        concepts.extend(proper_nouns)
        
        # 提取常见医学术语模式
        medical_patterns = [
            r'\b\w+itis\b',  # 炎症
            r'\b\w+osis\b',  # 病变
            r'\b\w+emia\b',  # 血液
            r'\b\w+pathy\b', # 病理
            r'\b\w+oma\b',   # 肿瘤
        ]
        for pattern in medical_patterns:
            matches = re.findall(pattern, question_lower)
            concepts.extend(matches)
        
        return list(set(concepts))


class EvidenceFocuser:
    """
    证据聚焦：从检索到的passages中提取与问题最相关的句子。
    """
    
    def focus_evidence(
        self, 
        question: str, 
        passages: List[str], 
        question_entities: List[str],
        max_sentences: int = 10
    ) -> str:
        """
        从passages中提取最相关的句子，构建聚焦的证据。
        """
        all_sentences = []
        
        for passage in passages:
            # 分句
            sentences = re.split(r'(?<=[.!?])\s+', passage)
            for sent in sentences:
                sent = sent.strip()
                if len(sent) < 20:
                    continue
                
                # 计算句子与问题的相关性
                relevance = self._compute_relevance(sent, question, question_entities)
                all_sentences.append((sent, relevance))
        
        # 按相关性排序
        all_sentences.sort(key=lambda x: x[1], reverse=True)
        
        # 选择top句子
        focused_sentences = [sent for sent, score in all_sentences[:max_sentences]]
        
        return " ".join(focused_sentences)
    
    def _compute_relevance(
        self, 
        sentence: str, 
        question: str, 
        entities: List[str]
    ) -> float:
        """计算句子与问题的相关性分数。"""
        sent_lower = sentence.lower()
        question_lower = question.lower()
        
        # 1. 实体匹配分数
        entity_score = sum(1 for e in entities if e.lower() in sent_lower) / max(len(entities), 1)
        
        # 2. 关键词重叠分数
        q_words = set(re.findall(r'\b\w{4,}\b', question_lower))
        s_words = set(re.findall(r'\b\w{4,}\b', sent_lower))
        overlap = len(q_words & s_words) / max(len(q_words), 1)
        
        # 3. 结论性句子加分
        conclusion_indicators = ['conclude', 'result', 'found', 'show', 'demonstrate', 'indicate']
        conclusion_bonus = 0.2 if any(ind in sent_lower for ind in conclusion_indicators) else 0
        
        return entity_score * 0.4 + overlap * 0.4 + conclusion_bonus * 0.2


class OptionContrastiveRetriever:
    """
    选项对比检索：为MCQ的每个选项找最强支持证据，对比选项间差异。
    """
    
    def __init__(self, embedding_model):
        self.embedding_model = embedding_model
    
    def retrieve_for_options(
        self,
        question: str,
        options: Dict[str, str],
        passages: List[str],
        passage_embeddings: np.ndarray,
        top_k_per_option: int = 3
    ) -> Tuple[Dict[str, List[str]], str]:
        """
        为每个选项检索最相关的passages。
        
        Returns:
            option_evidence: 每个选项的支持证据
            best_option: 证据最强的选项
        """
        option_evidence = {}
        option_scores = {}
        
        for opt_key, opt_text in options.items():
            # 构建选项查询：问题 + 选项
            query = f"{question} {opt_text}"
            query_embedding = self.embedding_model.encode(
                query, normalize_embeddings=True, show_progress_bar=False
            )
            
            # 计算与所有passages的相似度
            similarities = np.dot(passage_embeddings, query_embedding)
            
            # 选择top-k
            top_indices = np.argsort(similarities)[::-1][:top_k_per_option]
            top_passages = [passages[i] for i in top_indices]
            top_scores = similarities[top_indices]
            
            option_evidence[opt_key] = top_passages
            option_scores[opt_key] = float(np.mean(top_scores))
        
        # 找证据最强的选项
        best_option = max(option_scores, key=option_scores.get)
        
        return option_evidence, best_option, option_scores
    
    def build_contrastive_context(
        self,
        question: str,
        options: Dict[str, str],
        option_evidence: Dict[str, List[str]],
        best_option: str
    ) -> str:
        """
        构建对比性上下文，突出最佳选项的证据。
        """
        context = "[Evidence for answer options]\n\n"
        
        # 首先展示最佳选项的证据
        context += f"[Most relevant evidence (supports option {best_option})]\n"
        for i, passage in enumerate(option_evidence[best_option][:2], 1):
            context += f"{i}. {passage[:300]}...\n\n"
        
        # 简要展示其他选项的证据
        for opt_key in options:
            if opt_key != best_option and option_evidence.get(opt_key):
                context += f"[Evidence related to option {opt_key}]\n"
                context += f"- {option_evidence[opt_key][0][:200]}...\n\n"
        
        return context


class BidirectionalRetriever:
    """
    双向检索：同时检索支持和反对的证据，提供平衡视角。
    用于Yes/No/Maybe类问题。
    """
    
    def __init__(self, embedding_model):
        self.embedding_model = embedding_model
    
    def retrieve_bidirectional(
        self,
        question: str,
        passages: List[str],
        passage_embeddings: np.ndarray,
        top_k: int = 5
    ) -> Tuple[List[str], List[str], str]:
        """
        双向检索：分别检索支持和反对的证据。
        
        Returns:
            supporting_evidence: 支持性证据
            opposing_evidence: 反对性证据
            recommendation: 'yes', 'no', or 'maybe'
        """
        # 构建支持性查询
        support_query = f"{question} Yes, this is true. Evidence supports."
        support_embedding = self.embedding_model.encode(
            support_query, normalize_embeddings=True, show_progress_bar=False
        )
        
        # 构建反对性查询
        oppose_query = f"{question} No, this is false. Evidence contradicts."
        oppose_embedding = self.embedding_model.encode(
            oppose_query, normalize_embeddings=True, show_progress_bar=False
        )
        
        # 计算相似度
        support_scores = np.dot(passage_embeddings, support_embedding)
        oppose_scores = np.dot(passage_embeddings, oppose_embedding)
        
        # 选择支持性证据
        support_indices = np.argsort(support_scores)[::-1][:top_k]
        supporting = [passages[i] for i in support_indices]
        support_strength = float(np.mean(support_scores[support_indices]))
        
        # 选择反对性证据
        oppose_indices = np.argsort(oppose_scores)[::-1][:top_k]
        opposing = [passages[i] for i in oppose_indices]
        oppose_strength = float(np.mean(oppose_scores[oppose_indices]))
        
        # 判断推荐答案
        diff = support_strength - oppose_strength
        if diff > 0.05:
            recommendation = 'yes'
        elif diff < -0.05:
            recommendation = 'no'
        else:
            recommendation = 'maybe'
        
        return supporting, opposing, recommendation
    
    def build_balanced_context(
        self,
        question: str,
        supporting: List[str],
        opposing: List[str],
        recommendation: str
    ) -> str:
        """
        构建平衡的上下文，展示双方证据。
        """
        context = "[Scientific Evidence Analysis]\n\n"
        
        context += "[Evidence that MAY SUPPORT the claim]\n"
        for i, passage in enumerate(supporting[:3], 1):
            context += f"{i}. {passage[:250]}...\n"
        context += "\n"
        
        context += "[Evidence that MAY CONTRADICT the claim]\n"
        for i, passage in enumerate(opposing[:2], 1):
            context += f"{i}. {passage[:250]}...\n"
        context += "\n"
        
        context += f"[Preliminary assessment: {recommendation.upper()}]\n"
        
        return context


class MultiHopReasoner:
    """
    多跳推理：利用超图的实体关系进行多步推理。
    """
    
    def __init__(self, hypergraph_store):
        self.hypergraph_store = hypergraph_store
    
    def expand_reasoning_path(
        self,
        seed_entities: List[str],
        question: str,
        max_hops: int = 2
    ) -> List[Dict]:
        """
        从种子实体出发，通过超边扩展推理路径。
        
        Returns:
            reasoning_paths: 推理路径列表
        """
        paths = []
        visited = set(seed_entities)
        
        current_entities = seed_entities
        for hop in range(max_hops):
            next_entities = []
            
            for entity in current_entities:
                # 获取包含该实体的超边
                hyperedges = self.hypergraph_store.get_hyperedges_by_entity(entity)
                
                for he_id in hyperedges[:5]:  # 限制每个实体扩展的超边数
                    he_score = self.hypergraph_store.get_hyperedge_score(he_id)
                    if he_score < 0.4:
                        continue
                    
                    # 获取超边中的共现实体
                    co_entities = self.hypergraph_store.get_entities_by_hyperedge(he_id)
                    he_text = self.hypergraph_store.get_hyperedge_text(he_id)
                    
                    # 记录推理路径
                    path = {
                        'from_entity': entity,
                        'hyperedge': he_text,
                        'to_entities': [e for e in co_entities if e not in visited],
                        'score': he_score,
                        'hop': hop + 1
                    }
                    paths.append(path)
                    
                    # 添加新实体到下一轮
                    for e in co_entities:
                        if e not in visited:
                            visited.add(e)
                            next_entities.append(e)
            
            current_entities = next_entities[:20]  # 限制扩展规模
        
        return paths
    
    def build_reasoning_context(self, paths: List[Dict], max_paths: int = 5) -> str:
        """
        从推理路径构建上下文。
        """
        if not paths:
            return ""
        
        # 按分数排序
        paths.sort(key=lambda x: x['score'], reverse=True)
        
        context = "[Medical Knowledge Connections]\n"
        for i, path in enumerate(paths[:max_paths], 1):
            context += f"{i}. {path['hyperedge'][:200]}\n"
        
        return context


class RetrievalQualityAssessor:
    """
    检索质量评估：判断检索结果是否足够好。
    """
    
    def assess_quality(
        self,
        question: str,
        passages: List[str],
        question_entities: List[str]
    ) -> Tuple[float, str]:
        """
        评估检索质量。
        
        Returns:
            quality_score: 0-1的质量分数
            assessment: 'good', 'moderate', 'poor'
        """
        if not passages:
            return 0.0, 'poor'
        
        question_lower = question.lower()
        q_words = set(re.findall(r'\b\w{4,}\b', question_lower))
        
        scores = []
        for passage in passages[:5]:
            passage_lower = passage.lower()
            p_words = set(re.findall(r'\b\w{4,}\b', passage_lower))
            
            # 词重叠
            word_overlap = len(q_words & p_words) / max(len(q_words), 1)
            
            # 实体匹配
            entity_match = sum(1 for e in question_entities if e.lower() in passage_lower)
            entity_score = entity_match / max(len(question_entities), 1)
            
            scores.append(word_overlap * 0.5 + entity_score * 0.5)
        
        quality = np.mean(scores) if scores else 0
        
        if quality > 0.4:
            return quality, 'good'
        elif quality > 0.2:
            return quality, 'moderate'
        else:
            return quality, 'poor'
    
    def should_use_retrieval(
        self,
        quality_score: float,
        dataset: str
    ) -> bool:
        """
        判断是否应该使用检索结果。
        """
        thresholds = {
            'mmlu': 0.25,      # MMLU对检索依赖较低
            'medqa': 0.15,    # MedQA需要更多检索
            'medmcqa': 0.15,
            'pubmedqa': 0.2,
            'bioasq': 0.2,
        }
        threshold = thresholds.get(dataset.lower(), 0.2)
        return quality_score >= threshold
