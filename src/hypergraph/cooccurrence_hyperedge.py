"""
Sentence-level entity co-occurrence hyperedge builder.
Zero LLM token cost - fully local processing.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Set, Optional, Tuple
import hashlib
import logging

logger = logging.getLogger(__name__)


@dataclass
class Hyperedge:
    """
    A hyperedge representing an n-ary relation between multiple entities.
    
    Attributes:
        text: Natural language description (the source sentence)
        entities: List of entity names connected by this hyperedge
        score: Confidence score (0-1), based on entity count and type patterns
        hash_id: Unique identifier for this hyperedge
        entity_types: Optional dict mapping entity names to their types
    """
    text: str
    entities: List[str]
    score: float = 1.0
    hash_id: str = ""
    entity_types: Dict[str, str] = field(default_factory=dict)
    
    def __post_init__(self):
        if not self.hash_id:
            self.hash_id = self._generate_hash_id()
    
    def _generate_hash_id(self) -> str:
        """Generate unique hash ID from text content"""
        content = f"{self.text}|{'|'.join(sorted(self.entities))}"
        return hashlib.md5(content.encode()).hexdigest()[:16]
    
    def __hash__(self):
        return hash(self.hash_id)
    
    def __eq__(self, other):
        if isinstance(other, Hyperedge):
            return self.hash_id == other.hash_id
        return False
    
    def to_dict(self) -> Dict:
        return {
            "text": self.text,
            "entities": self.entities,
            "score": self.score,
            "hash_id": self.hash_id,
            "entity_types": self.entity_types,
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> "Hyperedge":
        return cls(
            text=data["text"],
            entities=data["entities"],
            score=data.get("score", 1.0),
            hash_id=data.get("hash_id", ""),
            entity_types=data.get("entity_types", {}),
        )


class CooccurrenceHyperedgeBuilder:
    """
    Build hyperedges based on sentence-level entity co-occurrence.
    
    This approach:
    - Reuses existing NER results (sentence_to_entities)
    - Zero additional computation cost
    - Naturally preserves sentence context as hyperedge description
    
    Usage:
        builder = CooccurrenceHyperedgeBuilder(min_entities=2)
        hyperedges = builder.build_from_ner_results(sentence_to_entities)
    """
    
    def __init__(
        self,
        min_entities: int = 2,
        max_entities: int = 10,
        min_sentence_length: int = 20,
        max_sentence_length: int = 500,
    ):
        """
        Initialize the hyperedge builder.
        
        Args:
            min_entities: Minimum entities required to form a hyperedge
            max_entities: Maximum entities per hyperedge (for filtering noise)
            min_sentence_length: Minimum sentence length to consider
            max_sentence_length: Maximum sentence length to consider
        """
        self.min_entities = min_entities
        self.max_entities = max_entities
        self.min_sentence_length = min_sentence_length
        self.max_sentence_length = max_sentence_length
    
    def build_from_ner_results(
        self,
        sentence_to_entities: Dict[str, Set[str]],
        entity_types: Optional[Dict[str, str]] = None,
    ) -> List[Hyperedge]:
        """
        Build hyperedges from NER results.
        
        Args:
            sentence_to_entities: Mapping from sentence text to set of entities
            entity_types: Optional mapping from entity name to entity type
        
        Returns:
            List of Hyperedge objects
        """
        if not sentence_to_entities:
            logger.warning("No sentence_to_entities provided, returning empty list")
            return []
        
        entity_types = entity_types or {}
        hyperedges = []
        
        # Calculate max entities for normalization
        valid_counts = [
            len(ents) for sent, ents in sentence_to_entities.items()
            if self._is_valid_sentence(sent, ents)
        ]
        max_entity_count = max(valid_counts) if valid_counts else 1
        
        for sentence, entities in sentence_to_entities.items():
            if not self._is_valid_sentence(sentence, entities):
                continue
            
            # Convert to list and ensure lowercase for consistency
            entity_list = [e.lower().strip() for e in entities if e.strip()]
            entity_list = list(set(entity_list))  # Deduplicate
            
            if len(entity_list) < self.min_entities:
                continue
            
            # Calculate base score based on entity count
            base_score = len(entity_list) / max_entity_count
            
            # Get entity types for this hyperedge
            he_entity_types = {
                e: entity_types.get(e, "UNKNOWN") 
                for e in entity_list
            }
            
            hyperedge = Hyperedge(
                text=sentence.strip(),
                entities=entity_list,
                score=base_score,
                entity_types=he_entity_types,
            )
            hyperedges.append(hyperedge)
        
        logger.info(f"Built {len(hyperedges)} hyperedges from {len(sentence_to_entities)} sentences")
        return hyperedges
    
    def _is_valid_sentence(self, sentence: str, entities: Set[str]) -> bool:
        """Check if sentence is valid for hyperedge construction"""
        if not sentence or not entities:
            return False
        
        sent_len = len(sentence)
        if sent_len < self.min_sentence_length or sent_len > self.max_sentence_length:
            return False
        
        entity_count = len(entities)
        if entity_count < self.min_entities or entity_count > self.max_entities:
            return False
        
        return True
    
    def build_from_passage_sentences(
        self,
        passage_hash_id_to_entities: Dict[str, Set[str]],
        sentence_to_entities: Dict[str, Set[str]],
        passage_hash_id_to_text: Dict[str, str],
        entity_types: Optional[Dict[str, str]] = None,
    ) -> Tuple[List[Hyperedge], Dict[str, List[str]]]:
        """
        Build hyperedges and track passage-hyperedge mappings.
        
        Returns:
            Tuple of (hyperedges, passage_to_hyperedge_ids mapping)
        """
        hyperedges = self.build_from_ner_results(sentence_to_entities, entity_types)
        
        # Build passage to hyperedge mapping
        passage_to_hyperedge_ids = {}
        hyperedge_text_to_id = {he.text: he.hash_id for he in hyperedges}
        
        for passage_hash_id, passage_text in passage_hash_id_to_text.items():
            matching_hyperedge_ids = []
            for he in hyperedges:
                # Check if hyperedge sentence is in passage
                if he.text in passage_text:
                    matching_hyperedge_ids.append(he.hash_id)
            
            if matching_hyperedge_ids:
                passage_to_hyperedge_ids[passage_hash_id] = matching_hyperedge_ids
        
        return hyperedges, passage_to_hyperedge_ids


class MedicalHyperedgeEnhancer:
    """
    Enhance hyperedge scores based on medical domain patterns.
    
    This applies domain-specific heuristics to boost hyperedges
    that represent clinically relevant relationships.
    """
    
    # Medical relation patterns with score boost factors
    MEDICAL_RELATION_PATTERNS = [
        # Diagnosis pattern: Symptom + Disease
        ({"SYMPTOM", "DISEASE"}, 1.2),
        ({"SIGN", "DISEASE"}, 1.2),
        
        # Treatment pattern: Disease + Drug/Treatment
        ({"DISEASE", "CHEMICAL"}, 1.3),
        ({"DISEASE", "DRUG"}, 1.3),
        ({"DISEASE", "TREATMENT"}, 1.3),
        
        # Lab/Diagnostic pattern
        ({"LAB", "VALUE", "DIAGNOSIS"}, 1.5),
        ({"LAB_TEST", "DISEASE"}, 1.3),
        ({"DIAGNOSTIC_PROCEDURE", "DISEASE"}, 1.3),
        
        # Mechanism pattern: Drug + Biological process
        ({"CHEMICAL", "GENE"}, 1.2),
        ({"DRUG", "PROTEIN"}, 1.2),
        ({"CHEMICAL", "PATHWAY"}, 1.2),
        
        # Anatomy + Disease/Symptom
        ({"ANATOMY", "DISEASE"}, 1.1),
        ({"BODY_PART", "SYMPTOM"}, 1.1),
        
        # Risk factor pattern
        ({"RISK_FACTOR", "DISEASE"}, 1.2),
        
        # Procedure + Condition
        ({"PROCEDURE", "DISEASE"}, 1.2),
        ({"SURGICAL_PROCEDURE", "ANATOMY"}, 1.2),
    ]
    
    # Keyword-based type inference for entities without explicit types
    TYPE_INFERENCE_KEYWORDS = {
        "SYMPTOM": ["pain", "ache", "fever", "fatigue", "nausea", "vomiting", 
                    "cough", "dyspnea", "diarrhea", "headache", "weakness"],
        "DISEASE": ["disease", "syndrome", "disorder", "cancer", "carcinoma",
                    "infection", "itis", "osis", "pathy"],
        "CHEMICAL": ["drug", "medication", "therapy", "treatment", "cillin",
                     "mycin", "zole", "prazole", "sartan", "olol"],
        "ANATOMY": ["kidney", "heart", "liver", "lung", "brain", "bone",
                    "artery", "vein", "nerve", "muscle"],
        "LAB_TEST": ["level", "count", "test", "assay", "measurement"],
    }
    
    def __init__(self, max_boost: float = 1.5):
        """
        Initialize enhancer.
        
        Args:
            max_boost: Maximum score boost factor
        """
        self.max_boost = max_boost
    
    def enhance_hyperedges(
        self,
        hyperedges: List[Hyperedge],
        entity_types: Optional[Dict[str, str]] = None,
    ) -> List[Hyperedge]:
        """
        Enhance scores for all hyperedges based on medical patterns.
        
        Args:
            hyperedges: List of hyperedges to enhance
            entity_types: Optional global entity type mapping
        
        Returns:
            List of hyperedges with enhanced scores
        """
        entity_types = entity_types or {}
        
        for he in hyperedges:
            # Get entity types (from hyperedge or global mapping or inference)
            type_set = self._get_entity_types(he, entity_types)
            
            # Calculate boost based on patterns
            boost = self._calculate_boost(type_set)
            
            # Apply boost (capped by max_boost)
            he.score = min(he.score * boost, self.max_boost)
        
        return hyperedges
    
    def _get_entity_types(
        self, 
        hyperedge: Hyperedge, 
        global_types: Dict[str, str]
    ) -> Set[str]:
        """Get entity types for a hyperedge, with inference fallback"""
        types = set()
        
        for entity in hyperedge.entities:
            # First check hyperedge's own type mapping
            if entity in hyperedge.entity_types:
                types.add(hyperedge.entity_types[entity])
            # Then check global mapping
            elif entity in global_types:
                types.add(global_types[entity])
            # Finally, try keyword-based inference
            else:
                inferred = self._infer_type(entity)
                if inferred:
                    types.add(inferred)
        
        return types
    
    def _infer_type(self, entity: str) -> Optional[str]:
        """Infer entity type based on keywords"""
        entity_lower = entity.lower()
        
        for type_name, keywords in self.TYPE_INFERENCE_KEYWORDS.items():
            for keyword in keywords:
                if keyword in entity_lower:
                    return type_name
        
        return None
    
    def _calculate_boost(self, type_set: Set[str]) -> float:
        """Calculate score boost based on entity type patterns"""
        max_boost = 1.0
        
        for pattern, boost in self.MEDICAL_RELATION_PATTERNS:
            if pattern.issubset(type_set):
                max_boost = max(max_boost, boost)
        
        return max_boost
    
    def enhance_score(
        self, 
        hyperedge: Hyperedge, 
        entity_types: Optional[Dict[str, str]] = None
    ) -> float:
        """
        Calculate enhanced score for a single hyperedge.
        
        Args:
            hyperedge: The hyperedge to score
            entity_types: Optional entity type mapping
        
        Returns:
            Enhanced score value
        """
        entity_types = entity_types or {}
        type_set = self._get_entity_types(hyperedge, entity_types)
        boost = self._calculate_boost(type_set)
        return min(hyperedge.score * boost, self.max_boost)

