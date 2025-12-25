from copy import deepcopy
from src.utils import compute_mdhash_id
import numpy as np
import pandas as pd
import os
import torch

class EmbeddingStore:
    def __init__(self, embedding_model, db_filename, batch_size=32, namespace="", model_type='passage'):
        """
        Initialize EmbeddingStore
        
        Args:
            embedding_model: SentenceTransformer model
            db_filename: Path to parquet file
            batch_size: Batch size for embedding (reduce if OOM)
            namespace: Namespace for hash generation
        """
        self.embedding_model = embedding_model
        self.db_filename = db_filename
        self.batch_size = batch_size
        self.namespace = namespace
        
        # Initialize storage attributes
        self.texts = []
        self.hash_ids = []
        self.embeddings = []
        self.text_to_hash_id = {}
        self.hash_id_to_text = {}
        self.hash_id_to_idx = {}
        
        # Load existing data if available
        self.load_from_parquet()
    
    def insert_text(self, texts):
        """Insert texts with memory-efficient batching"""
        if not texts:
            print(f"[{self.namespace}] No texts to insert")
            return
        
        print(f"[{self.namespace}] Inserting {len(texts)} texts into {self.namespace}...")
        
        # Create set from existing texts
        existing_texts_set = set(self.text_to_hash_id.keys())
        
        # Filter out duplicates
        new_texts = [text for text in texts if text not in existing_texts_set]
        
        if not new_texts:
            print(f"[{self.namespace}] No new texts to insert (all already exist)")
            return
        
        print(f"[{self.namespace}] Encoding {len(new_texts)} new texts...")
        
        # Process in smaller batches to avoid OOM
        all_embeddings = []
        num_batches = (len(new_texts) + self.batch_size - 1) // self.batch_size
        
        for i in range(0, len(new_texts), self.batch_size):
            batch = new_texts[i:i + self.batch_size]
            batch_num = i // self.batch_size + 1
            
            if batch_num % 1000 == 0 or batch_num == num_batches:
                print(f" Batch {batch_num}/{num_batches}...")
            
            try:
                # Clear CUDA cache before each batch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
                batch_embeddings = self.embedding_model.encode(
                    batch,
                    normalize_embeddings=True,
                    show_progress_bar=False,
                    batch_size=min(self.batch_size, 16),
                    convert_to_numpy=True
                )
                all_embeddings.append(batch_embeddings)
                
            except RuntimeError as e:
                if "out of memory" in str(e):
                    print(f"\n⚠️  OOM at batch {batch_num}, processing individually...")
                    torch.cuda.empty_cache()
                    
                    # Retry with individual items
                    batch_embeddings = []
                    for text in batch:
                        emb = self.embedding_model.encode(
                            [text],
                            normalize_embeddings=True,
                            show_progress_bar=False
                        )
                        batch_embeddings.append(emb[0])
                    all_embeddings.append(np.array(batch_embeddings))
                else:
                    raise
        
        new_embeddings = np.vstack(all_embeddings)
        
        # Generate hash IDs
        import hashlib
        new_hash_ids = []
        for text in new_texts:
            hash_object = hashlib.sha256(f"{self.namespace}-{text}".encode())
            hash_id = f"{self.namespace}-{hash_object.hexdigest()}"
            new_hash_ids.append(hash_id)
        
        # Update internal storage
        start_idx = len(self.texts)
        self.texts.extend(new_texts)
        self.hash_ids.extend(new_hash_ids)
        self.embeddings.extend(new_embeddings.tolist())
        
        # Update mappings
        for idx, (text, hash_id) in enumerate(zip(new_texts, new_hash_ids)):
            self.text_to_hash_id[text] = hash_id
            self.hash_id_to_text[hash_id] = text
            self.hash_id_to_idx[hash_id] = start_idx + idx
        
        # Save to disk
        self.save_to_parquet()
        
        print(f"[{self.namespace}] ✅ Inserted {len(new_texts)} new texts, total: {len(self.texts)}")
    
    def load_from_parquet(self):
        """Load existing data from parquet file"""
        if not os.path.exists(self.db_filename):
            print(f"[{self.namespace}] No existing data found at {self.db_filename}")
            return
        
        try:
            df = pd.read_parquet(self.db_filename)
            
            self.texts = df['text'].tolist()
            self.hash_ids = df['hash_id'].tolist()
            self.embeddings = df['embedding'].apply(lambda x: np.array(x)).tolist()
            
            # Rebuild mappings
            for idx, (text, hash_id) in enumerate(zip(self.texts, self.hash_ids)):
                self.text_to_hash_id[text] = hash_id
                self.hash_id_to_idx[hash_id] = idx
                self.hash_id_to_text[hash_id] = text
            
            print(f"[{self.namespace}] ✅ Loaded {len(self.texts)} existing texts")
        except Exception as e:
            print(f"[{self.namespace}] ⚠️  Error loading: {e}")
            self.texts = []
            self.hash_ids = []
            self.embeddings = []
            self.text_to_hash_id = {}
            self.hash_id_to_idx = {}
            self.hash_id_to_text = {}
    
    def save_to_parquet(self):
        """Save current data to parquet file"""
        # Create directory if needed
        os.makedirs(os.path.dirname(self.db_filename), exist_ok=True)
        
        # Convert to DataFrame
        df = pd.DataFrame({
            'text': self.texts,
            'hash_id': self.hash_ids,
            'embedding': [emb.tolist() if isinstance(emb, np.ndarray) else emb 
                         for emb in self.embeddings]
        })
        
        # Save
        df.to_parquet(self.db_filename, index=False)
    
    def get_hash_id_to_text(self):
        """Get mapping from hash_id to text"""
        return self.hash_id_to_text