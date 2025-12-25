"""
Hypergraph storage using bipartite graph structure.
Enables efficient querying of entity-hyperedge relationships.
"""

import os
import json
import pickle
import logging
from typing import Dict, List, Set, Optional, Tuple, Any
from collections import defaultdict
from dataclasses import dataclass, field

import numpy as np

try:
    import igraph as ig
except ImportError:
    ig = None

from .cooccurrence_hyperedge import Hyperedge

logger = logging.getLogger(__name__)


@dataclass
class HypergraphStats:
    """Statistics about the hypergraph"""
    num_entities: int = 0
    num_hyperedges: int = 0
    num_edges: int = 0
    avg_entities_per_hyperedge: float = 0.0
    avg_hyperedges_per_entity: float = 0.0


class HypergraphStore:
    """
    Hypergraph storage using bipartite graph structure.
    
    The hypergraph G_H = (V, E_H) is stored as a bipartite graph G_B = (V_B, E_B):
    - V_B = V ∪ E_H (entity nodes + hyperedge nodes)
    - E_B = {(e_H, v) | e_H ∈ E_H, v ∈ V_eH}
    
    This allows:
    - Query all entities in a hyperedge: neighbors of hyperedge node
    - Query all hyperedges containing an entity: neighbors of entity node
    - Efficient graph database operations
    
    Usage:
        store = HypergraphStore(storage_dir)
        store.add_hyperedges(hyperedges)
        entities = store.get_entities_by_hyperedge(hyperedge_id)
        hyperedges = store.get_hyperedges_by_entity(entity_id)
    """
    
    def __init__(self, storage_dir: str):
        """
        Initialize hypergraph store.
        
        Args:
            storage_dir: Directory to store hypergraph data
        """
        self.storage_dir = storage_dir
        os.makedirs(storage_dir, exist_ok=True)
        
        # Node storage
        self.entity_to_idx: Dict[str, int] = {}  # entity hash_id -> node index
        self.hyperedge_to_idx: Dict[str, int] = {}  # hyperedge hash_id -> node index
        self.idx_to_entity: Dict[int, str] = {}
        self.idx_to_hyperedge: Dict[int, str] = {}
        
        # Text storage
        self.entity_hash_to_text: Dict[str, str] = {}  # entity hash_id -> entity text
        self.hyperedge_hash_to_text: Dict[str, str] = {}  # hyperedge hash_id -> sentence text
        
        # Hyperedge metadata
        self.hyperedge_scores: Dict[str, float] = {}  # hyperedge hash_id -> score
        self.hyperedge_entities: Dict[str, List[str]] = {}  # hyperedge hash_id -> entity list
        
        # Adjacency lists (for fast queries without graph library)
        self.entity_to_hyperedges: Dict[str, Set[str]] = defaultdict(set)
        self.hyperedge_to_entities: Dict[str, Set[str]] = defaultdict(set)
        
        # Optional igraph for PPR
        self.graph: Optional[Any] = None
        self._node_count = 0
        
        # File paths
        self.metadata_path = os.path.join(storage_dir, "hypergraph_metadata.json")
        self.adjacency_path = os.path.join(storage_dir, "hypergraph_adjacency.pkl")
        self.graph_path = os.path.join(storage_dir, "hypergraph.graphml")
    
    def add_hyperedges(
        self, 
        hyperedges: List[Hyperedge],
        entity_hash_fn: Optional[callable] = None,
    ) -> None:
        """
        Add hyperedges to the store.
        
        Args:
            hyperedges: List of Hyperedge objects
            entity_hash_fn: Optional function to generate entity hash IDs
        """
        import hashlib
        
        if entity_hash_fn is None:
            def entity_hash_fn(text):
                return hashlib.md5(text.lower().encode()).hexdigest()[:16]
        
        for he in hyperedges:
            hyperedge_id = he.hash_id
            
            # Store hyperedge metadata
            self.hyperedge_hash_to_text[hyperedge_id] = he.text
            self.hyperedge_scores[hyperedge_id] = he.score
            self.hyperedge_entities[hyperedge_id] = he.entities
            
            # Add hyperedge node if new
            if hyperedge_id not in self.hyperedge_to_idx:
                idx = self._node_count
                self.hyperedge_to_idx[hyperedge_id] = idx
                self.idx_to_hyperedge[idx] = hyperedge_id
                self._node_count += 1
            
            # Process entities
            entity_ids = []
            for entity_text in he.entities:
                entity_id = entity_hash_fn(entity_text)
                entity_ids.append(entity_id)
                
                # Store entity text
                self.entity_hash_to_text[entity_id] = entity_text
                
                # Add entity node if new
                if entity_id not in self.entity_to_idx:
                    idx = self._node_count
                    self.entity_to_idx[entity_id] = idx
                    self.idx_to_entity[idx] = entity_id
                    self._node_count += 1
                
                # Add adjacency links
                self.entity_to_hyperedges[entity_id].add(hyperedge_id)
                self.hyperedge_to_entities[hyperedge_id].add(entity_id)
        
        logger.info(f"Added {len(hyperedges)} hyperedges, "
                   f"total entities: {len(self.entity_to_idx)}, "
                   f"total hyperedges: {len(self.hyperedge_to_idx)}")
    
    def get_entities_by_hyperedge(self, hyperedge_id: str) -> List[str]:
        """
        Get all entity IDs connected to a hyperedge.
        
        Args:
            hyperedge_id: Hyperedge hash ID
        
        Returns:
            List of entity hash IDs
        """
        return list(self.hyperedge_to_entities.get(hyperedge_id, set()))
    
    def get_hyperedges_by_entity(self, entity_id: str) -> List[str]:
        """
        Get all hyperedge IDs containing an entity.
        
        Args:
            entity_id: Entity hash ID
        
        Returns:
            List of hyperedge hash IDs
        """
        return list(self.entity_to_hyperedges.get(entity_id, set()))
    
    def get_hyperedge_text(self, hyperedge_id: str) -> Optional[str]:
        """Get the text description of a hyperedge"""
        return self.hyperedge_hash_to_text.get(hyperedge_id)
    
    def get_entity_text(self, entity_id: str) -> Optional[str]:
        """Get the text of an entity"""
        return self.entity_hash_to_text.get(entity_id)
    
    def get_hyperedge_score(self, hyperedge_id: str) -> float:
        """Get the confidence score of a hyperedge"""
        return self.hyperedge_scores.get(hyperedge_id, 0.0)
    
    def get_all_hyperedge_ids(self) -> List[str]:
        """Get all hyperedge IDs"""
        return list(self.hyperedge_to_idx.keys())
    
    def get_all_entity_ids(self) -> List[str]:
        """Get all entity IDs"""
        return list(self.entity_to_idx.keys())
    
    def get_stats(self) -> HypergraphStats:
        """Get statistics about the hypergraph"""
        num_entities = len(self.entity_to_idx)
        num_hyperedges = len(self.hyperedge_to_idx)
        num_edges = sum(len(entities) for entities in self.hyperedge_to_entities.values())
        
        avg_entities_per_he = (
            num_edges / num_hyperedges if num_hyperedges > 0 else 0
        )
        avg_hyperedges_per_entity = (
            num_edges / num_entities if num_entities > 0 else 0
        )
        
        return HypergraphStats(
            num_entities=num_entities,
            num_hyperedges=num_hyperedges,
            num_edges=num_edges,
            avg_entities_per_hyperedge=avg_entities_per_he,
            avg_hyperedges_per_entity=avg_hyperedges_per_entity,
        )
    
    def build_igraph(self) -> Optional[Any]:
        """
        Build igraph representation for PPR and graph algorithms.
        
        Returns:
            igraph.Graph object or None if igraph not available
        """
        if ig is None:
            logger.warning("igraph not available, skipping graph build")
            return None
        
        self.graph = ig.Graph(directed=False)
        
        # Add all nodes
        all_nodes = []
        node_types = []
        node_names = []
        
        # Add entity nodes first
        for entity_id, idx in sorted(self.entity_to_idx.items(), key=lambda x: x[1]):
            all_nodes.append(entity_id)
            node_types.append("entity")
            node_names.append(self.entity_hash_to_text.get(entity_id, entity_id))
        
        # Add hyperedge nodes
        for he_id, idx in sorted(self.hyperedge_to_idx.items(), key=lambda x: x[1]):
            all_nodes.append(he_id)
            node_types.append("hyperedge")
            node_names.append(self.hyperedge_hash_to_text.get(he_id, he_id)[:100])
        
        self.graph.add_vertices(len(all_nodes))
        self.graph.vs["hash_id"] = all_nodes
        self.graph.vs["type"] = node_types
        self.graph.vs["name"] = node_names
        
        # Build node ID to graph index mapping
        hash_to_graph_idx = {h: i for i, h in enumerate(all_nodes)}
        
        # Add edges
        edges = []
        weights = []
        
        for he_id, entity_ids in self.hyperedge_to_entities.items():
            he_graph_idx = hash_to_graph_idx.get(he_id)
            if he_graph_idx is None:
                continue
            
            he_score = self.hyperedge_scores.get(he_id, 1.0)
            
            for entity_id in entity_ids:
                entity_graph_idx = hash_to_graph_idx.get(entity_id)
                if entity_graph_idx is None:
                    continue
                
                edges.append((he_graph_idx, entity_graph_idx))
                weights.append(he_score)
        
        self.graph.add_edges(edges)
        self.graph.es["weight"] = weights
        
        logger.info(f"Built igraph with {self.graph.vcount()} nodes and {self.graph.ecount()} edges")
        return self.graph
    
    def save(self) -> None:
        """Save hypergraph to disk"""
        # Save metadata
        metadata = {
            "entity_to_idx": self.entity_to_idx,
            "hyperedge_to_idx": self.hyperedge_to_idx,
            "entity_hash_to_text": self.entity_hash_to_text,
            "hyperedge_hash_to_text": self.hyperedge_hash_to_text,
            "hyperedge_scores": self.hyperedge_scores,
            "hyperedge_entities": self.hyperedge_entities,
            "_node_count": self._node_count,
        }
        
        with open(self.metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)
        
        # Save adjacency lists
        adjacency = {
            "entity_to_hyperedges": {k: list(v) for k, v in self.entity_to_hyperedges.items()},
            "hyperedge_to_entities": {k: list(v) for k, v in self.hyperedge_to_entities.items()},
        }
        
        with open(self.adjacency_path, "wb") as f:
            pickle.dump(adjacency, f)
        
        # Save igraph if available
        if self.graph is not None:
            self.graph.write_graphml(self.graph_path)
        
        logger.info(f"Saved hypergraph to {self.storage_dir}")
    
    def load(self) -> bool:
        """
        Load hypergraph from disk.
        
        Returns:
            True if loaded successfully, False otherwise
        """
        if not os.path.exists(self.metadata_path):
            logger.warning(f"No hypergraph metadata found at {self.metadata_path}")
            return False
        
        try:
            # Load metadata
            with open(self.metadata_path, "r") as f:
                metadata = json.load(f)
            
            self.entity_to_idx = metadata["entity_to_idx"]
            self.hyperedge_to_idx = metadata["hyperedge_to_idx"]
            self.entity_hash_to_text = metadata["entity_hash_to_text"]
            self.hyperedge_hash_to_text = metadata["hyperedge_hash_to_text"]
            self.hyperedge_scores = metadata["hyperedge_scores"]
            self.hyperedge_entities = metadata["hyperedge_entities"]
            self._node_count = metadata["_node_count"]
            
            # Rebuild reverse mappings
            self.idx_to_entity = {int(v): k for k, v in self.entity_to_idx.items()}
            self.idx_to_hyperedge = {int(v): k for k, v in self.hyperedge_to_idx.items()}
            
            # Load adjacency lists
            if os.path.exists(self.adjacency_path):
                with open(self.adjacency_path, "rb") as f:
                    adjacency = pickle.load(f)
                
                self.entity_to_hyperedges = defaultdict(
                    set, {k: set(v) for k, v in adjacency["entity_to_hyperedges"].items()}
                )
                self.hyperedge_to_entities = defaultdict(
                    set, {k: set(v) for k, v in adjacency["hyperedge_to_entities"].items()}
                )
            
            # Load igraph if available
            if ig is not None and os.path.exists(self.graph_path):
                self.graph = ig.Graph.Read_GraphML(self.graph_path)
            
            logger.info(f"Loaded hypergraph from {self.storage_dir}")
            return True
            
        except Exception as e:
            logger.error(f"Error loading hypergraph: {e}")
            return False
    
    def merge_with_linear_graph(
        self, 
        linear_graph: Any,
        passage_to_hyperedge_ids: Dict[str, List[str]],
    ) -> Any:
        """
        Merge hypergraph with LinearRAG's entity-passage graph.
        
        This creates a unified graph with:
        - Original entity and passage nodes
        - New hyperedge nodes
        - Connections: entity-passage, entity-hyperedge, passage-hyperedge
        
        Args:
            linear_graph: The original LinearRAG igraph
            passage_to_hyperedge_ids: Mapping from passage hash IDs to hyperedge IDs
        
        Returns:
            Merged igraph
        """
        if ig is None:
            logger.warning("igraph not available")
            return linear_graph
        
        # Get existing nodes from LinearRAG graph
        existing_names = set(linear_graph.vs["name"])
        
        # Add hyperedge nodes that don't exist
        new_vertices = []
        for he_id in self.hyperedge_to_idx.keys():
            if he_id not in existing_names:
                new_vertices.append(he_id)
        
        if new_vertices:
            linear_graph.add_vertices(len(new_vertices))
            # Set attributes for new vertices
            start_idx = linear_graph.vcount() - len(new_vertices)
            for i, he_id in enumerate(new_vertices):
                idx = start_idx + i
                linear_graph.vs[idx]["name"] = he_id
                linear_graph.vs[idx]["content"] = self.hyperedge_hash_to_text.get(he_id, "")[:200]
                linear_graph.vs[idx]["type"] = "hyperedge"
        
        # Build name to index mapping
        name_to_idx = {v["name"]: v.index for v in linear_graph.vs}
        
        # Add edges: passage -> hyperedge
        new_edges = []
        new_weights = []
        
        for passage_id, he_ids in passage_to_hyperedge_ids.items():
            passage_idx = name_to_idx.get(passage_id)
            if passage_idx is None:
                continue
            
            for he_id in he_ids:
                he_idx = name_to_idx.get(he_id)
                if he_idx is None:
                    continue
                
                new_edges.append((passage_idx, he_idx))
                new_weights.append(self.hyperedge_scores.get(he_id, 1.0))
        
        # Add edges: entity -> hyperedge (for entities that exist in both)
        for entity_id, he_ids in self.entity_to_hyperedges.items():
            entity_idx = name_to_idx.get(entity_id)
            if entity_idx is None:
                continue
            
            for he_id in he_ids:
                he_idx = name_to_idx.get(he_id)
                if he_idx is None:
                    continue
                
                new_edges.append((entity_idx, he_idx))
                new_weights.append(self.hyperedge_scores.get(he_id, 1.0) * 0.8)
        
        if new_edges:
            linear_graph.add_edges(new_edges)
            # Extend weights
            existing_weights = linear_graph.es["weight"] if "weight" in linear_graph.es.attributes() else []
            linear_graph.es["weight"] = list(existing_weights) + new_weights
        
        logger.info(f"Merged hypergraph: added {len(new_vertices)} hyperedge nodes, "
                   f"{len(new_edges)} edges")
        
        return linear_graph

