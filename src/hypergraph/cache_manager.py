"""
Multi-level caching system for HyperLinearRAG.
Caches NER results, embeddings, and hypergraph structures.
"""

import os
import json
import pickle
import hashlib
import logging
from typing import Dict, List, Set, Optional, Any, Tuple
from dataclasses import dataclass
from abc import ABC, abstractmethod

import numpy as np

try:
    import pandas as pd
    import pyarrow.parquet as pq
    HAS_PARQUET = True
except ImportError:
    HAS_PARQUET = False

logger = logging.getLogger(__name__)


@dataclass
class CacheStats:
    """Cache statistics"""
    hits: int = 0
    misses: int = 0
    size: int = 0
    
    @property
    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0


class CacheLevel(ABC):
    """Abstract base class for cache levels"""
    
    @abstractmethod
    def get(self, key: str) -> Optional[Any]:
        pass
    
    @abstractmethod
    def set(self, key: str, value: Any) -> None:
        pass
    
    @abstractmethod
    def exists(self, key: str) -> bool:
        pass
    
    @abstractmethod
    def clear(self) -> None:
        pass


class NERCache(CacheLevel):
    """
    Level 1: NER results cache.
    Stores entity extraction results keyed by document hash.
    """
    
    def __init__(self, cache_path: str):
        self.cache_path = cache_path
        self.cache: Dict[str, Dict] = {}
        self.stats = CacheStats()
        self._load()
    
    def _load(self) -> None:
        """Load cache from disk"""
        if os.path.exists(self.cache_path):
            try:
                with open(self.cache_path, "rb") as f:
                    self.cache = pickle.load(f)
                self.stats.size = len(self.cache)
                logger.info(f"Loaded NER cache with {self.stats.size} entries")
            except Exception as e:
                logger.warning(f"Failed to load NER cache: {e}")
                self.cache = {}
    
    def _save(self) -> None:
        """Save cache to disk"""
        os.makedirs(os.path.dirname(self.cache_path), exist_ok=True)
        with open(self.cache_path, "wb") as f:
            pickle.dump(self.cache, f)
    
    def get(self, doc_hash: str) -> Optional[Dict]:
        """
        Get NER results for a document.
        
        Args:
            doc_hash: Document content hash
        
        Returns:
            Dict with 'passage_entities' and 'sentence_to_entities' or None
        """
        if doc_hash in self.cache:
            self.stats.hits += 1
            return self.cache[doc_hash]
        
        self.stats.misses += 1
        return None
    
    def set(self, doc_hash: str, value: Dict) -> None:
        """
        Store NER results for a document.
        
        Args:
            doc_hash: Document content hash
            value: Dict with 'passage_entities' and 'sentence_to_entities'
        """
        self.cache[doc_hash] = value
        self.stats.size = len(self.cache)
    
    def set_batch(self, items: Dict[str, Dict]) -> None:
        """Store multiple NER results at once"""
        self.cache.update(items)
        self.stats.size = len(self.cache)
    
    def exists(self, doc_hash: str) -> bool:
        return doc_hash in self.cache
    
    def clear(self) -> None:
        self.cache = {}
        self.stats = CacheStats()
        if os.path.exists(self.cache_path):
            os.remove(self.cache_path)
    
    def save(self) -> None:
        """Persist cache to disk"""
        self._save()
        logger.info(f"Saved NER cache with {self.stats.size} entries")


class EmbeddingCache(CacheLevel):
    """
    Level 2: Embedding vectors cache.
    Stores text embeddings keyed by text hash.
    Uses parquet for efficient storage and random access.
    """
    
    def __init__(self, cache_path: str, embedding_dim: int = 768):
        self.cache_path = cache_path
        self.embedding_dim = embedding_dim
        self.stats = CacheStats()
        
        # In-memory index: text_hash -> row_index
        self.index: Dict[str, int] = {}
        
        # In-memory cache for frequent access
        self.memory_cache: Dict[str, np.ndarray] = {}
        self.memory_cache_max_size = 10000
        
        # Load index if exists
        self._load_index()
    
    def _load_index(self) -> None:
        """Load embedding index from disk"""
        index_path = self.cache_path + ".index.json"
        if os.path.exists(index_path):
            try:
                with open(index_path, "r") as f:
                    self.index = json.load(f)
                self.stats.size = len(self.index)
                logger.info(f"Loaded embedding cache index with {self.stats.size} entries")
            except Exception as e:
                logger.warning(f"Failed to load embedding index: {e}")
    
    def _save_index(self) -> None:
        """Save embedding index to disk"""
        index_path = self.cache_path + ".index.json"
        os.makedirs(os.path.dirname(self.cache_path), exist_ok=True)
        with open(index_path, "w") as f:
            json.dump(self.index, f)
    
    def get(self, text_hash: str) -> Optional[np.ndarray]:
        """
        Get embedding for a text.
        
        Args:
            text_hash: Hash of the text
        
        Returns:
            Embedding vector or None
        """
        # Check memory cache first
        if text_hash in self.memory_cache:
            self.stats.hits += 1
            return self.memory_cache[text_hash]
        
        # Check disk cache
        if text_hash not in self.index:
            self.stats.misses += 1
            return None
        
        # Load from disk (if parquet available)
        if HAS_PARQUET and os.path.exists(self.cache_path):
            try:
                row_idx = self.index[text_hash]
                table = pq.read_table(self.cache_path, columns=["embedding"])
                embedding = np.array(table["embedding"][row_idx].as_py())
                
                # Add to memory cache
                if len(self.memory_cache) < self.memory_cache_max_size:
                    self.memory_cache[text_hash] = embedding
                
                self.stats.hits += 1
                return embedding
            except Exception as e:
                logger.warning(f"Failed to load embedding: {e}")
        
        self.stats.misses += 1
        return None
    
    def set(self, text_hash: str, embedding: np.ndarray) -> None:
        """
        Store embedding for a text.
        
        Args:
            text_hash: Hash of the text
            embedding: Embedding vector
        """
        # Add to memory cache
        if len(self.memory_cache) < self.memory_cache_max_size:
            self.memory_cache[text_hash] = embedding
        
        # Update index
        if text_hash not in self.index:
            self.index[text_hash] = len(self.index)
            self.stats.size = len(self.index)
    
    def set_batch(self, items: Dict[str, np.ndarray]) -> None:
        """Store multiple embeddings at once"""
        for text_hash, embedding in items.items():
            self.set(text_hash, embedding)
    
    def exists(self, text_hash: str) -> bool:
        return text_hash in self.index or text_hash in self.memory_cache
    
    def clear(self) -> None:
        self.index = {}
        self.memory_cache = {}
        self.stats = CacheStats()
        
        if os.path.exists(self.cache_path):
            os.remove(self.cache_path)
        index_path = self.cache_path + ".index.json"
        if os.path.exists(index_path):
            os.remove(index_path)
    
    def save(self) -> None:
        """Persist embeddings to disk"""
        if not HAS_PARQUET:
            logger.warning("Parquet not available, using pickle fallback")
            fallback_path = self.cache_path + ".pkl"
            with open(fallback_path, "wb") as f:
                pickle.dump(self.memory_cache, f)
            self._save_index()
            return
        
        if not self.memory_cache:
            return
        
        # Convert to DataFrame and save as parquet
        data = {
            "text_hash": list(self.memory_cache.keys()),
            "embedding": [emb.tolist() for emb in self.memory_cache.values()],
        }
        df = pd.DataFrame(data)
        
        os.makedirs(os.path.dirname(self.cache_path), exist_ok=True)
        df.to_parquet(self.cache_path, index=False)
        self._save_index()
        
        logger.info(f"Saved embedding cache with {len(self.memory_cache)} entries")


class HypergraphCache(CacheLevel):
    """
    Level 3: Hypergraph structure cache.
    Stores hyperedges and entity mappings.
    """
    
    def __init__(self, cache_path: str):
        self.cache_path = cache_path
        self.cache: Dict[str, Any] = {
            "hyperedges": {},  # hyperedge_hash -> hyperedge_data
            "entity_to_hyperedges": {},  # entity_hash -> [hyperedge_hashes]
            "hyperedge_to_entities": {},  # hyperedge_hash -> [entity_hashes]
        }
        self.stats = CacheStats()
        self._load()
    
    def _load(self) -> None:
        """Load cache from disk"""
        if os.path.exists(self.cache_path):
            try:
                with open(self.cache_path, "rb") as f:
                    self.cache = pickle.load(f)
                self.stats.size = len(self.cache.get("hyperedges", {}))
                logger.info(f"Loaded hypergraph cache with {self.stats.size} hyperedges")
            except Exception as e:
                logger.warning(f"Failed to load hypergraph cache: {e}")
    
    def get(self, key: str) -> Optional[Any]:
        """Get hyperedge data by hash"""
        if key in self.cache["hyperedges"]:
            self.stats.hits += 1
            return self.cache["hyperedges"][key]
        self.stats.misses += 1
        return None
    
    def get_hyperedges_by_entity(self, entity_hash: str) -> List[str]:
        """Get hyperedge hashes for an entity"""
        return self.cache["entity_to_hyperedges"].get(entity_hash, [])
    
    def get_entities_by_hyperedge(self, hyperedge_hash: str) -> List[str]:
        """Get entity hashes for a hyperedge"""
        return self.cache["hyperedge_to_entities"].get(hyperedge_hash, [])
    
    def set(self, hyperedge_hash: str, hyperedge_data: Dict) -> None:
        """Store hyperedge data"""
        self.cache["hyperedges"][hyperedge_hash] = hyperedge_data
        self.stats.size = len(self.cache["hyperedges"])
    
    def set_mappings(
        self, 
        entity_to_hyperedges: Dict[str, List[str]],
        hyperedge_to_entities: Dict[str, List[str]],
    ) -> None:
        """Set entity-hyperedge mappings"""
        self.cache["entity_to_hyperedges"] = entity_to_hyperedges
        self.cache["hyperedge_to_entities"] = hyperedge_to_entities
    
    def exists(self, hyperedge_hash: str) -> bool:
        return hyperedge_hash in self.cache["hyperedges"]
    
    def clear(self) -> None:
        self.cache = {
            "hyperedges": {},
            "entity_to_hyperedges": {},
            "hyperedge_to_entities": {},
        }
        self.stats = CacheStats()
        if os.path.exists(self.cache_path):
            os.remove(self.cache_path)
    
    def save(self) -> None:
        """Persist cache to disk"""
        os.makedirs(os.path.dirname(self.cache_path), exist_ok=True)
        with open(self.cache_path, "wb") as f:
            pickle.dump(self.cache, f)
        logger.info(f"Saved hypergraph cache with {self.stats.size} hyperedges")


class MultiLevelCache:
    """
    Multi-level cache manager for HyperLinearRAG.
    
    Levels:
    1. NER Cache: Document hash -> NER results
    2. Embedding Cache: Text hash -> Embedding vector
    3. Hypergraph Cache: Hyperedge structures and mappings
    
    Usage:
        cache = MultiLevelCache(cache_dir)
        
        # NER caching
        ner_result = cache.get_ner(doc_hash)
        cache.set_ner(doc_hash, ner_result)
        
        # Embedding caching
        embedding = cache.get_embedding(text_hash)
        cache.set_embedding(text_hash, embedding)
        
        # Persist all
        cache.save_all()
    """
    
    def __init__(self, cache_dir: str, embedding_dim: int = 768):
        """
        Initialize multi-level cache.
        
        Args:
            cache_dir: Directory for cache files
            embedding_dim: Embedding vector dimension
        """
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        
        # Initialize cache levels
        self.ner_cache = NERCache(os.path.join(cache_dir, "ner_cache.pkl"))
        self.embedding_cache = EmbeddingCache(
            os.path.join(cache_dir, "embedding_cache.parquet"),
            embedding_dim=embedding_dim
        )
        self.hypergraph_cache = HypergraphCache(
            os.path.join(cache_dir, "hypergraph_cache.pkl")
        )
    
    # NER Cache methods
    def get_ner(self, doc_hash: str) -> Optional[Dict]:
        """Get NER results from cache"""
        return self.ner_cache.get(doc_hash)
    
    def set_ner(self, doc_hash: str, value: Dict) -> None:
        """Store NER results in cache"""
        self.ner_cache.set(doc_hash, value)
    
    def has_ner(self, doc_hash: str) -> bool:
        """Check if NER results exist"""
        return self.ner_cache.exists(doc_hash)
    
    # Embedding Cache methods
    def get_embedding(self, text_hash: str) -> Optional[np.ndarray]:
        """Get embedding from cache"""
        return self.embedding_cache.get(text_hash)
    
    def set_embedding(self, text_hash: str, embedding: np.ndarray) -> None:
        """Store embedding in cache"""
        self.embedding_cache.set(text_hash, embedding)
    
    def has_embedding(self, text_hash: str) -> bool:
        """Check if embedding exists"""
        return self.embedding_cache.exists(text_hash)
    
    # Hypergraph Cache methods
    def get_hyperedge(self, hyperedge_hash: str) -> Optional[Dict]:
        """Get hyperedge data from cache"""
        return self.hypergraph_cache.get(hyperedge_hash)
    
    def set_hyperedge(self, hyperedge_hash: str, data: Dict) -> None:
        """Store hyperedge in cache"""
        self.hypergraph_cache.set(hyperedge_hash, data)
    
    def get_hyperedges_by_entity(self, entity_hash: str) -> List[str]:
        """Get hyperedges containing an entity"""
        return self.hypergraph_cache.get_hyperedges_by_entity(entity_hash)
    
    # Batch operations
    def set_ner_batch(self, items: Dict[str, Dict]) -> None:
        """Store multiple NER results"""
        self.ner_cache.set_batch(items)
    
    def set_embedding_batch(self, items: Dict[str, np.ndarray]) -> None:
        """Store multiple embeddings"""
        self.embedding_cache.set_batch(items)
    
    # Persistence
    def save_all(self) -> None:
        """Persist all cache levels to disk"""
        self.ner_cache.save()
        self.embedding_cache.save()
        self.hypergraph_cache.save()
        logger.info(f"Saved all caches to {self.cache_dir}")
    
    def clear_all(self) -> None:
        """Clear all cache levels"""
        self.ner_cache.clear()
        self.embedding_cache.clear()
        self.hypergraph_cache.clear()
        logger.info("Cleared all caches")
    
    def get_stats(self) -> Dict[str, CacheStats]:
        """Get statistics for all cache levels"""
        return {
            "ner": self.ner_cache.stats,
            "embedding": self.embedding_cache.stats,
            "hypergraph": self.hypergraph_cache.stats,
        }
    
    @staticmethod
    def compute_text_hash(text: str) -> str:
        """Compute hash for text content"""
        return hashlib.md5(text.encode()).hexdigest()

