"""
Incremental indexing for HyperLinearRAG.
Only process new/modified documents, avoiding redundant computation.
"""

import os
import json
import hashlib
import logging
from datetime import datetime
from typing import Dict, List, Set, Tuple, Optional, Any
from dataclasses import dataclass, field, asdict

logger = logging.getLogger(__name__)


@dataclass
class IndexManifest:
    """Manifest tracking indexed documents"""
    version: str = "1.0"
    created_at: str = ""
    updated_at: str = ""
    doc_hashes: Dict[str, str] = field(default_factory=dict)  # doc_hash -> timestamp
    entity_count: int = 0
    hyperedge_count: int = 0
    passage_count: int = 0
    
    def __post_init__(self):
        if not self.created_at:
            self.created_at = datetime.now().isoformat()
        self.updated_at = datetime.now().isoformat()


class IncrementalIndexer:
    """
    Incremental indexing manager for HyperLinearRAG.
    
    Features:
    - Track indexed documents via content hashes
    - Identify new/modified documents
    - Support incremental updates to NER, embeddings, and hypergraph
    
    Usage:
        indexer = IncrementalIndexer(index_dir)
        new_docs, existing_hashes = indexer.get_new_documents(documents)
        # Process only new_docs
        indexer.update_index(new_ner_results, new_hyperedges, stats)
    """
    
    def __init__(self, index_dir: str):
        """
        Initialize incremental indexer.
        
        Args:
            index_dir: Directory containing index files
        """
        self.index_dir = index_dir
        os.makedirs(index_dir, exist_ok=True)
        
        self.manifest_path = os.path.join(index_dir, "index_manifest.json")
        self.manifest = self._load_manifest()
    
    def _load_manifest(self) -> IndexManifest:
        """Load or create index manifest"""
        if os.path.exists(self.manifest_path):
            try:
                with open(self.manifest_path, "r") as f:
                    data = json.load(f)
                return IndexManifest(**data)
            except Exception as e:
                logger.warning(f"Failed to load manifest: {e}, creating new one")
        
        return IndexManifest()
    
    def _save_manifest(self) -> None:
        """Save manifest to disk"""
        self.manifest.updated_at = datetime.now().isoformat()
        with open(self.manifest_path, "w") as f:
            json.dump(asdict(self.manifest), f, indent=2)
    
    @staticmethod
    def compute_doc_hash(text: str) -> str:
        """Compute hash for document content"""
        return hashlib.md5(text.encode()).hexdigest()
    
    def get_new_documents(
        self, 
        documents: List[str]
    ) -> Tuple[List[str], List[str], Set[str]]:
        """
        Identify new documents that haven't been indexed.
        
        Args:
            documents: List of document texts
        
        Returns:
            Tuple of (new_documents, new_doc_hashes, existing_hashes)
        """
        existing_hashes = set(self.manifest.doc_hashes.keys())
        
        new_docs = []
        new_hashes = []
        
        for doc in documents:
            doc_hash = self.compute_doc_hash(doc)
            if doc_hash not in existing_hashes:
                new_docs.append(doc)
                new_hashes.append(doc_hash)
        
        logger.info(f"Found {len(new_docs)} new documents out of {len(documents)} total")
        return new_docs, new_hashes, existing_hashes
    
    def get_indexed_doc_hashes(self) -> Set[str]:
        """Get set of already indexed document hashes"""
        return set(self.manifest.doc_hashes.keys())
    
    def mark_documents_indexed(
        self, 
        doc_hashes: List[str],
        stats: Optional[Dict[str, int]] = None,
    ) -> None:
        """
        Mark documents as indexed.
        
        Args:
            doc_hashes: List of document hashes that were indexed
            stats: Optional statistics to update (entity_count, hyperedge_count, etc.)
        """
        timestamp = datetime.now().isoformat()
        
        for doc_hash in doc_hashes:
            self.manifest.doc_hashes[doc_hash] = timestamp
        
        if stats:
            if "entity_count" in stats:
                self.manifest.entity_count = stats["entity_count"]
            if "hyperedge_count" in stats:
                self.manifest.hyperedge_count = stats["hyperedge_count"]
            if "passage_count" in stats:
                self.manifest.passage_count = stats["passage_count"]
        
        self._save_manifest()
        logger.info(f"Marked {len(doc_hashes)} documents as indexed")
    
    def update_index(
        self,
        new_doc_hashes: List[str],
        entity_count: int = 0,
        hyperedge_count: int = 0,
        passage_count: int = 0,
    ) -> None:
        """
        Update index with new documents.
        
        Args:
            new_doc_hashes: Hashes of newly indexed documents
            entity_count: Total entity count after update
            hyperedge_count: Total hyperedge count after update
            passage_count: Total passage count after update
        """
        self.mark_documents_indexed(
            new_doc_hashes,
            stats={
                "entity_count": entity_count,
                "hyperedge_count": hyperedge_count,
                "passage_count": passage_count,
            }
        )
    
    def get_stats(self) -> Dict[str, Any]:
        """Get indexing statistics"""
        return {
            "version": self.manifest.version,
            "created_at": self.manifest.created_at,
            "updated_at": self.manifest.updated_at,
            "indexed_documents": len(self.manifest.doc_hashes),
            "entity_count": self.manifest.entity_count,
            "hyperedge_count": self.manifest.hyperedge_count,
            "passage_count": self.manifest.passage_count,
        }
    
    def clear_index(self) -> None:
        """Clear all index data (for full rebuild)"""
        self.manifest = IndexManifest()
        self._save_manifest()
        logger.info("Cleared index manifest")
    
    def needs_rebuild(self, min_docs: int = 0) -> bool:
        """
        Check if index needs full rebuild.
        
        Args:
            min_docs: Minimum documents threshold
        
        Returns:
            True if rebuild is recommended
        """
        indexed_count = len(self.manifest.doc_hashes)
        return indexed_count < min_docs


class BatchDocumentProcessor:
    """
    Process documents in batches for efficient incremental indexing.
    """
    
    def __init__(
        self, 
        batch_size: int = 100,
        incremental_indexer: Optional[IncrementalIndexer] = None,
    ):
        """
        Initialize batch processor.
        
        Args:
            batch_size: Number of documents per batch
            incremental_indexer: Optional incremental indexer for tracking
        """
        self.batch_size = batch_size
        self.indexer = incremental_indexer
    
    def process_documents(
        self,
        documents: List[str],
        process_fn: callable,
        skip_indexed: bool = True,
    ) -> List[Any]:
        """
        Process documents in batches.
        
        Args:
            documents: List of documents to process
            process_fn: Function to process a batch of documents
            skip_indexed: Whether to skip already indexed documents
        
        Returns:
            Combined results from all batches
        """
        # Filter to new documents if incremental
        if skip_indexed and self.indexer:
            new_docs, new_hashes, _ = self.indexer.get_new_documents(documents)
            documents_to_process = new_docs
            doc_hashes = new_hashes
        else:
            documents_to_process = documents
            doc_hashes = [IncrementalIndexer.compute_doc_hash(d) for d in documents]
        
        if not documents_to_process:
            logger.info("No new documents to process")
            return []
        
        logger.info(f"Processing {len(documents_to_process)} documents in batches of {self.batch_size}")
        
        all_results = []
        processed_hashes = []
        
        for i in range(0, len(documents_to_process), self.batch_size):
            batch = documents_to_process[i:i + self.batch_size]
            batch_hashes = doc_hashes[i:i + self.batch_size]
            
            # Process batch
            results = process_fn(batch)
            all_results.extend(results)
            processed_hashes.extend(batch_hashes)
            
            # Update indexer after each batch
            if self.indexer:
                self.indexer.mark_documents_indexed(batch_hashes)
            
            logger.info(f"Processed batch {i // self.batch_size + 1}, "
                       f"total: {len(processed_hashes)}/{len(documents_to_process)}")
        
        return all_results

