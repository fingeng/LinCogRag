"""
Dataset-adaptive retrieval strategies for LinCogRAG.
Implements evidence decisiveness scoring and specialized retrieval for different datasets.
"""

from typing import List, Dict, Tuple, Optional, Set
import numpy as np
import re
from collections import defaultdict
import logging

logger = logging.getLogger(__name__)


class EvidenceDecisivenessScorer:
    """
    Evaluate how decisive a passage is for answering a question.
    Decisive evidence clearly supports or refutes a claim, unlike neutral/vague content.
    """
    
    # Strong positive evidence indicators
    DECISIVE_POSITIVE = [
        "clearly", "definitively", "proven", "established", "demonstrated",
        "strongly associated", "significant effect", "significantly", 
        "recommended", "first-line", "gold standard", "treatment of choice",
        "confirms", "shows that", "evidence supports", "effective for",
        "superior to", "preferred", "approved for"
    ]
    
    # Strong negative evidence indicators  
    DECISIVE_NEGATIVE = [
        "no evidence", "not recommended", "contraindicated", "avoid",
        "ineffective", "no significant", "failed to show", "no benefit",
        "should not", "not effective", "rejected", "disproven",
        "no association", "no correlation", "not associated"
    ]
    
    # Uncertainty indicators (reduce decisiveness)
    UNCERTAIN_INDICATORS = [
        "may", "might", "possibly", "unclear", "conflicting",
        "limited evidence", "further research", "controversial",
        "some studies", "mixed results", "inconclusive", "uncertain",
        "preliminary", "suggested", "potential", "appears to"
    ]
    
    def score_decisiveness(self, passage: str, question: str = "") -> float:
        """
        Calculate decisiveness score (0-1).
        Higher score = more decisive evidence.
        """
        text = passage.lower()
        
        pos_count = sum(1 for ind in self.DECISIVE_POSITIVE if ind in text)
        neg_count = sum(1 for ind in self.DECISIVE_NEGATIVE if ind in text)
        unc_count = sum(1 for ind in self.UNCERTAIN_INDICATORS if ind in text)
        
        total = pos_count + neg_count + unc_count
        if total == 0:
            return 0.5  # Neutral
        
        # Decisiveness = (positive + negative indicators) / total
        decisiveness = (pos_count + neg_count) / total
        
        # Bonus for explicit conclusions
        if any(phrase in text for phrase in ["in conclusion", "we conclude", "the results show"]):
            decisiveness = min(1.0, decisiveness + 0.1)
        
        return decisiveness
    
    def get_evidence_direction(self, passage: str) -> str:
        """
        Determine evidence direction: 'positive', 'negative', or 'neutral'.
        """
        text = passage.lower()
        
        pos_count = sum(1 for ind in self.DECISIVE_POSITIVE if ind in text)
        neg_count = sum(1 for ind in self.DECISIVE_NEGATIVE if ind in text)
        
        if pos_count > neg_count + 1:
            return 'positive'
        elif neg_count > pos_count + 1:
            return 'negative'
        return 'neutral'
    
    def filter_decisive_passages(
        self, 
        passages: List[str], 
        question: str,
        min_decisiveness: float = 0.5,
        top_k: int = 5
    ) -> List[str]:
        """Filter and rank passages by decisiveness."""
        scored = []
        for p in passages:
            dec_score = self.score_decisiveness(p, question)
            direction = self.get_evidence_direction(p)
            scored.append((p, dec_score, direction))
        
        # Sort by decisiveness
        scored.sort(key=lambda x: x[1], reverse=True)
        
        # Filter by minimum threshold
        filtered = [(p, s, d) for p, s, d in scored if s >= min_decisiveness]
        
        if not filtered:
            return [p for p, s, d in scored[:top_k]]
        
        return [p for p, s, d in filtered[:top_k]]


class DatasetAdaptiveRetriever:
    """
    Adaptive retrieval strategies for different dataset types.
    """
    
    def __init__(self, embedding_model, decisiveness_scorer: Optional[EvidenceDecisivenessScorer] = None):
        self.embedding_model = embedding_model
        self.decisiveness_scorer = decisiveness_scorer or EvidenceDecisivenessScorer()
    
    def extract_options(self, question: str) -> Dict[str, str]:
        """
        从MCQ问题中提取选项。
        """
        options = {}
        
        # 匹配 (A) xxx 或 A. xxx 或 A) xxx 格式
        patterns = [
            r'\(([A-E])\)\s*([^(\n]+?)(?=\([A-E]\)|$)',  # (A) xxx
            r'([A-E])\.\s*([^A-E\n]+?)(?=[A-E]\.|$)',    # A. xxx
            r'([A-E])\)\s*([^A-E\n]+?)(?=[A-E]\)|$)',    # A) xxx
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, question, re.IGNORECASE)
            if matches and len(matches) >= 2:
                for key, value in matches:
                    options[key.upper()] = value.strip()
                break
        
        return options
    
    def get_strategy(self, dataset: str) -> str:
        """Get retrieval strategy name for dataset."""
        dataset_lower = dataset.lower()
        
        if dataset_lower in ['pubmedqa']:
            return 'decisive_evidence'
        elif dataset_lower in ['bioasq']:
            return 'high_confidence'
        elif dataset_lower in ['medqa', 'medmcqa']:
            return 'option_aware'
        elif dataset_lower in ['mmlu']:
            return 'knowledge_augment'
        return 'standard'
    
    def rerank_for_pubmedqa(
        self,
        question: str,
        passages: List[str],
        passage_scores: List[float],
        top_k: int = 5
    ) -> Tuple[List[str], List[float]]:
        """
        PubMedQA-specific reranking: prioritize POSITIVE/SUPPORTIVE evidence.
        
        Key insight: PubMedQA questions often have "Yes" as answer when there's
        supporting evidence. We should prioritize finding supportive evidence
        to help LLM make confident "Yes" decisions instead of defaulting to "Maybe".
        """
        scored_passages = []
        
        for passage, base_score in zip(passages, passage_scores):
            # Calculate decisiveness
            decisiveness = self.decisiveness_scorer.score_decisiveness(passage, question)
            direction = self.decisiveness_scorer.get_evidence_direction(passage)
            
            # 🔥 优化：更积极地boost正向证据
            if direction == 'positive':
                adjusted_score = base_score * (1 + 0.8 * decisiveness)  # 更大的boost
            elif direction == 'negative':
                adjusted_score = base_score * (1 + 0.3 * decisiveness)  # 较小的boost
            else:
                adjusted_score = base_score * (1 + 0.2 * decisiveness)  # 中性证据最小boost
            
            scored_passages.append({
                'passage': passage,
                'base_score': base_score,
                'decisiveness': decisiveness,
                'direction': direction,
                'final_score': adjusted_score
            })
        
        # Sort by final score
        scored_passages.sort(key=lambda x: x['final_score'], reverse=True)
        
        # 🔥 优化策略：优先选择正向证据
        positive_passages = [p for p in scored_passages if p['direction'] == 'positive']
        
        if len(positive_passages) >= 2:
            # 有足够正向证据时，主要返回正向证据
            selected = positive_passages[:top_k]
        else:
            # 否则返回最高分的
            selected = scored_passages[:top_k]
        
        if len(selected) < top_k:
            remaining = [p for p in scored_passages if p not in selected]
            selected.extend(remaining[:top_k - len(selected)])
        
        return (
            [p['passage'] for p in selected],
            [p['final_score'] for p in selected]
        )
    
    def rerank_for_bioasq(
        self,
        question: str,
        passages: List[str],
        passage_scores: List[float],
        top_k: int = 5
    ) -> Tuple[List[str], List[float]]:
        """
        BioASQ-specific reranking: BALANCED evidence selection.
        
        Key insight: BioASQ has 63.5% Yes and 36.5% No answers.
        We need to provide balanced evidence to help LLM make accurate decisions.
        """
        scored_passages = []
        
        for passage, base_score in zip(passages, passage_scores):
            decisiveness = self.decisiveness_scorer.score_decisiveness(passage, question)
            direction = self.decisiveness_scorer.get_evidence_direction(passage)
            
            # 🔥 平衡策略：对正向和负向证据给予相似的boost
            if direction == 'positive':
                adjusted_score = base_score * (1 + 0.5 * decisiveness)
            elif direction == 'negative':
                adjusted_score = base_score * (1 + 0.5 * decisiveness)  # 同等boost
            else:
                adjusted_score = base_score * (1 + 0.2 * decisiveness)  # 中性较弱
            
            scored_passages.append({
                'passage': passage,
                'decisiveness': decisiveness,
                'direction': direction,
                'final_score': adjusted_score
            })
        
        # Sort by adjusted score (decisiveness)
        scored_passages.sort(key=lambda x: x['final_score'], reverse=True)
        
        # 🔥 返回最具决定性的证据，不偏向任何方向
        return (
            [p['passage'] for p in scored_passages[:top_k]],
            [p['final_score'] for p in scored_passages[:top_k]]
        )
    
    def rerank_for_mcq(
        self,
        question: str,
        options: Dict[str, str],
        passages: List[str],
        passage_scores: List[float],
        top_k: int = 5
    ) -> Tuple[List[str], List[float], Optional[str]]:
        """
        MCQ-specific reranking: option-aware evidence selection.
        Returns passages and optionally the predicted best option.
        """
        if not options:
            return passages[:top_k], passage_scores[:top_k], None
        
        # Score each option's evidence support
        option_support = defaultdict(float)
        option_passages = defaultdict(list)
        
        for passage, score in zip(passages, passage_scores):
            passage_lower = passage.lower()
            
            for opt_key, opt_text in options.items():
                opt_lower = opt_text.lower()
                
                # Check if option text appears in passage
                if opt_lower in passage_lower or any(
                    word in passage_lower 
                    for word in opt_lower.split() 
                    if len(word) > 4
                ):
                    option_support[opt_key] += score
                    option_passages[opt_key].append((passage, score))
        
        # Find best supported option
        if option_support:
            best_option = max(option_support, key=option_support.get)
            
            # Return passages supporting the best option
            if option_passages[best_option]:
                best_passages = option_passages[best_option]
                best_passages.sort(key=lambda x: x[1], reverse=True)
                return (
                    [p for p, s in best_passages[:top_k]],
                    [s for p, s in best_passages[:top_k]],
                    best_option
                )
        
        # Fallback: return top passages by original score
        return passages[:top_k], passage_scores[:top_k], None
    
    def extract_options(self, question: str) -> Dict[str, str]:
        """Extract options from MCQ question text."""
        options = {}
        
        # Pattern: A. text or A) text
        pattern = r'([ABCD])[.\)]\s*(.+?)(?=(?:[ABCD][.\)])|$)'
        matches = re.findall(pattern, question, re.DOTALL)
        
        for letter, text in matches:
            options[letter] = text.strip()
        
        return options


class MMULKnowledgeAugmenter:
    """
    Special handling for MMLU questions which often require 
    textbook-style medical knowledge rather than clinical literature.
    """
    
    # Medical concept synonyms for query expansion
    MEDICAL_SYNONYMS = {
        'heart attack': ['myocardial infarction', 'MI', 'cardiac event'],
        'high blood pressure': ['hypertension', 'HTN', 'elevated BP'],
        'diabetes': ['diabetes mellitus', 'DM', 'hyperglycemia'],
        'stroke': ['cerebrovascular accident', 'CVA', 'brain infarction'],
        'kidney failure': ['renal failure', 'nephropathy', 'CKD'],
        'liver disease': ['hepatic disease', 'hepatopathy', 'cirrhosis'],
        'lung cancer': ['pulmonary carcinoma', 'bronchogenic carcinoma'],
        'breast cancer': ['mammary carcinoma', 'breast carcinoma'],
    }
    
    def expand_query(self, question: str) -> str:
        """Expand query with medical synonyms."""
        expanded = question
        question_lower = question.lower()
        
        for term, synonyms in self.MEDICAL_SYNONYMS.items():
            if term in question_lower:
                expanded += " " + " ".join(synonyms)
        
        return expanded
    
    def should_skip_retrieval(self, question: str, passages: List[str], threshold: float = 0.3) -> bool:
        """
        Determine if retrieval quality is too low for MMLU.
        If so, let LLM use internal knowledge instead.
        """
        if not passages:
            return True
        
        # Simple heuristic: check question-passage overlap
        question_words = set(question.lower().split())
        question_words -= {'the', 'a', 'an', 'is', 'are', 'what', 'which', 'how', 'of', 'in', 'to'}
        
        overlap_scores = []
        for passage in passages[:3]:
            passage_words = set(passage.lower().split())
            overlap = len(question_words & passage_words) / len(question_words) if question_words else 0
            overlap_scores.append(overlap)
        
        avg_overlap = sum(overlap_scores) / len(overlap_scores) if overlap_scores else 0
        return avg_overlap < threshold


class HypergraphConditionChecker:
    """
    Check condition completeness using hypergraph relationships.
    Medical questions often require multiple conditions to be satisfied.
    """
    
    def check_condition_coverage(
        self,
        question_entities: Set[str],
        hyperedge_entities: Set[str]
    ) -> float:
        """
        Calculate how well hyperedge entities cover question conditions.
        Returns coverage ratio (0-1).
        """
        if not question_entities:
            return 0.0
        
        q_entities_lower = {e.lower() for e in question_entities}
        he_entities_lower = {e.lower() for e in hyperedge_entities}
        
        intersection = q_entities_lower & he_entities_lower
        coverage = len(intersection) / len(q_entities_lower)
        
        return coverage
    
    def select_complete_hyperedges(
        self,
        question_entities: Set[str],
        hyperedges: List[Dict],  # List of {entities: Set[str], score: float, text: str}
        min_coverage: float = 0.5,
        top_k: int = 5
    ) -> List[Dict]:
        """
        Select hyperedges that provide best condition coverage.
        Uses greedy selection to maximize coverage.
        """
        scored_hyperedges = []
        
        for he in hyperedges:
            he_entities = he.get('entities', set())
            coverage = self.check_condition_coverage(question_entities, he_entities)
            base_score = he.get('score', 1.0)
            
            # Combined score: coverage * base_score
            combined_score = coverage * base_score
            
            if coverage >= min_coverage:
                scored_hyperedges.append({
                    **he,
                    'coverage': coverage,
                    'combined_score': combined_score
                })
        
        # Sort by combined score
        scored_hyperedges.sort(key=lambda x: x['combined_score'], reverse=True)
        
        # Greedy selection for maximum coverage
        selected = []
        covered_entities = set()
        
        for he in scored_hyperedges:
            if len(selected) >= top_k:
                break
            
            he_entities = {e.lower() for e in he.get('entities', set())}
            new_coverage = he_entities - covered_entities
            
            # Select if adds new coverage or has high score
            if new_coverage or he['combined_score'] > 0.7:
                selected.append(he)
                covered_entities.update(he_entities)
        
        return selected
