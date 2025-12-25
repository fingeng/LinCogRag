import spacy
from collections import defaultdict
import torch
import re

class SpacyNER:
    def __init__(self, model_name, use_hybrid=True):
        """
        Initialize NER with optional hybrid strategy
        Args:
            model_name: spaCy model name
            use_hybrid: Use both BC5CDR and HuggingFace NER
        """
        self.use_hybrid = use_hybrid
        
        # Load BC5CDR model (primary)
        print(f"[NER] 🔧 Requested model: {model_name}")
        print(f"[NER] 🔧 Using BC5CDR medical NER model for stability")
        
        try:
            self.nlp = spacy.load("en_ner_bc5cdr_md")
            print(f"[NER] ✅ Loaded BC5CDR model successfully")
            print(f"[NER] Model info: {self.nlp.meta['name']}")
            print(f"[NER] Available entity labels: {self.nlp.pipe_labels['ner']}")
        except Exception as e:
            print(f"[NER] ⚠️  Failed to load BC5CDR model: {e}")
            print(f"[NER] Falling back to: {model_name}")
            self.nlp = spacy.load(model_name)
        
        # Initialize HuggingFace NER if hybrid mode is enabled
        self.hf_ner = None
        if self.use_hybrid:
            print("[NER] 🔧 Initializing HuggingFace NER as supplement...")
            try:
                from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline
                print("[NER] Loading Hugging Face model...")
                
                model_path = "models/biomedical-ner-all"
                print(f"[NER] Using local model: {model_path}")
                
                # 🔧 FIX: Use max_length aggregation strategy for better subword handling
                self.hf_tokenizer = AutoTokenizer.from_pretrained(model_path)
                self.hf_model = AutoModelForTokenClassification.from_pretrained(model_path)
                
                if torch.cuda.is_available():
                    self.hf_model = self.hf_model.cuda()
                
                self.hf_ner = pipeline(
                    "ner",
                    model=self.hf_model,
                    tokenizer=self.hf_tokenizer,
                    aggregation_strategy="max",  # ✅ 使用 max 策略获得最好的合并效果
                    device=0 if torch.cuda.is_available() else -1
                )
                print(f"[NER] ✅ Loaded successfully on {'CUDA' if torch.cuda.is_available() else 'CPU'}")
                print(f"[NER] 📋 Using aggregation_strategy='max' for optimal subword merging")
                
            except Exception as e:
                print(f"[NER] ⚠️  Failed to load HuggingFace NER: {e}")
                self.hf_ner = None
                self.use_hybrid = False
        
        if self.use_hybrid and self.hf_ner:
            print("[NER] ✅ Hybrid NER enabled (BC5CDR + HF)")
        else:
            print("[NER] ✅ BC5CDR NER only")

        # 🔧 ADD: Medical entity patterns for better coverage
        self.medical_patterns = [
            # Drug names (case-insensitive)
            (r'\b[A-Z][a-z]+(?:cillin|mycin|oxacin|zole|prazole|sartan|olol)\b', 'DRUG'),
            # Diseases
            (r'\b(?:carcinoma|adenocarcinoma|lymphoma|leukemia|sarcoma)\b', 'DISEASE'),
            # Symptoms
            (r'\b(?:pain|fever|cough|nausea|vomiting|dyspnea|tachycardia)\b', 'SYMPTOM'),
            # Lab values
            (r'\b(?:hemoglobin|leukocyte|platelet|glucose|creatinine)\s+(?:count|level)\b', 'LAB'),
        ]
    
    def question_ner(self, text):
        """Extract entities from question text with fallback strategies"""
        entities = set()
        
        # Strategy 1: BC5CDR NER (primary)
        try:
            doc = self.nlp(text)
            for ent in doc.ents:
                if ent.label_ in ['CHEMICAL', 'DISEASE']:
                    entity_text = ent.text.lower().strip()
                    if len(entity_text) > 2:
                        entities.add(entity_text)
        except Exception as e:
            print(f"[NER] BC5CDR extraction failed: {e}")
        
        # Strategy 2: HuggingFace NER (supplement)
        if self.use_hybrid and self.hf_ner:
            try:
                hf_entities = self._extract_hf_entities(text)
                entities.update(hf_entities)
            except Exception as e:
                print(f"[NER] HF extraction failed: {e}")
        
        # Strategy 3: Medical keyword extraction (fallback)
        if len(entities) == 0:
            medical_keywords = self._extract_medical_keywords(text)
            entities.update(medical_keywords)
        
        return list(entities)
    
    def _extract_hf_entities(self, text):
        """
        Extract entities using HuggingFace NER model
        Uses max aggregation strategy which properly merges subwords
        """
        if not self.hf_ner:
            return set()
        
        try:
            # 🔧 FIX: Run NER with max aggregation (handles subwords automatically)
            results = self.hf_ner(text)
            
            entities = set()
            for entity in results:
                # With aggregation_strategy="max", entity['word'] is already merged
                entity_text = entity['word'].strip()
                entity_type = entity.get('entity_group', '')
                
                # 🔧 Clean entity text (remove any remaining artifacts)
                entity_text = entity_text.replace('##', '')  # BERT-style
                entity_text = entity_text.replace('Ġ', ' ')  # GPT-style
                entity_text = entity_text.replace('▁', '')   # SentencePiece
                entity_text = entity_text.strip().lower()
                
                # Filter by length and type
                if len(entity_text) > 2 and entity_type:
                    entities.add(entity_text)
            
            return entities
            
        except Exception as e:
            print(f"[NER] _extract_hf_entities error: {e}")
            return set()
    
    def _extract_medical_keywords(self, text):
        """Extract medical-related terms as fallback"""
        medical_patterns = [
            # Diseases
            r'\b(?:infection|syndrome|disease|disorder|cancer|tumor|carcinoma|adenocarcinoma|sarcoma)\b',
            r'\b(?:fever|pain|cough|nausea|vomiting|diarrhea|headache|fatigue|weakness)\b',
            r'\b(?:diabetes|hypertension|asthma|pneumonia|hepatitis|tuberculosis|meningitis)\b',
            r'\b(?:arthritis|dermatitis|nephritis|bronchitis|colitis|gastritis)\b',
            
            # Medications
            r'\b(?:cisplatin|carboplatin|azithromycin|metformin|insulin|aspirin|ibuprofen)\b',
            r'\b(?:drug|medication|therapy|treatment|antibiotic|chemotherapy|analgesic)\b',
            
            # Pathogens
            r'\b(?:virus|bacteria|bacterial|viral|fungal|parasitic|pathogen)\b',
            
            # Medical terms
            r'\b(?:cardiac|pulmonary|renal|hepatic|neurological|respiratory|gastrointestinal)\b',
        ]
        
        keywords = set()
        text_lower = text.lower()
        
        for pattern in medical_patterns:
            matches = re.findall(pattern, text_lower)
            keywords.update(matches)
        
        return keywords

    def batch_ner(self, hash_id_to_text, max_workers=8):
        """Process NER on a batch of texts with parallel processing"""
        from concurrent.futures import ThreadPoolExecutor
        from tqdm import tqdm
        
        print(f"[NER] Starting batch NER on {len(hash_id_to_text)} passages...")
        print(f"[NER] Using {max_workers} workers")
        
        passage_hash_id_to_entities = {}
        sentence_to_entities = {}
        
        def process_single_passage(item):
            hash_id, text = item
            try:
                doc = self.nlp(text)
                passage_entities = set()
                
                # BC5CDR entities
                for ent in doc.ents:
                    if ent.label_ in ['CHEMICAL', 'DISEASE']:
                        entity_text = ent.text.lower().strip()
                        if len(entity_text) > 2:
                            passage_entities.add(entity_text)
                
                # HF entities (if enabled)
                if self.use_hybrid and self.hf_ner:
                    try:
                        hf_entities = self._extract_hf_entities(text)
                        passage_entities.update(hf_entities)
                    except:
                        pass
                
                # Process sentences
                local_sentence_entities = {}
                for sent in doc.sents:
                    sent_text = sent.text.strip()
                    if len(sent_text) < 10:
                        continue
                    
                    sent_entities = set()
                    
                    # BC5CDR entities in sentence
                    for ent in sent.ents:
                        if ent.label_ in ['CHEMICAL', 'DISEASE']:
                            entity_text = ent.text.lower().strip()
                            if len(entity_text) > 2:
                                sent_entities.add(entity_text)
                    
                    # HF entities in sentence
                    if self.use_hybrid and self.hf_ner:
                        try:
                            hf_sent_entities = self._extract_hf_entities(sent_text)
                            sent_entities.update(hf_sent_entities)
                        except:
                            pass
                    
                    if sent_entities:
                        local_sentence_entities[sent_text] = sent_entities
                
                return hash_id, passage_entities, local_sentence_entities
                
            except Exception as e:
                print(f"[NER] Error processing passage {hash_id[:20]}...: {e}")
                return hash_id, set(), {}
        
        # Process in parallel
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            results = list(tqdm(
                executor.map(process_single_passage, hash_id_to_text.items()),
                total=len(hash_id_to_text),
                desc="NER Processing"
            ))
        
        # Aggregate results
        for hash_id, passage_entities, local_sentence_entities in results:
            if passage_entities:
                passage_hash_id_to_entities[hash_id] = passage_entities
            sentence_to_entities.update(local_sentence_entities)
        
        print(f"[NER] ✅ Processed {len(passage_hash_id_to_entities)} passages with entities")
        print(f"[NER] ✅ Extracted {len(sentence_to_entities)} sentences with entities")
        
        return passage_hash_id_to_entities, sentence_to_entities


class BatchGPUNER:
    """
    GPU-optimized batch NER processing.
    Maximizes throughput by processing texts in batches on GPU.
    
    Usage:
        batch_ner = BatchGPUNER(batch_size=32)
        results = batch_ner.batch_process(texts)
    """
    
    def __init__(
        self, 
        model_path: str = "models/biomedical-ner-all",
        batch_size: int = 32,
        max_length: int = 512,
        score_threshold: float = 0.5,
    ):
        """
        Initialize GPU batch NER.
        
        Args:
            model_path: Path to HuggingFace NER model
            batch_size: Number of texts per batch
            max_length: Maximum sequence length
            score_threshold: Minimum confidence score for entities
        """
        self.batch_size = batch_size
        self.max_length = max_length
        self.score_threshold = score_threshold
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        print(f"[BatchGPUNER] Initializing on {self.device}...")
        
        from transformers import AutoTokenizer, AutoModelForTokenClassification
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForTokenClassification.from_pretrained(model_path)
        self.model.to(self.device)
        self.model.eval()
        
        # Get label mapping
        self.id2label = self.model.config.id2label
        
        print(f"[BatchGPUNER] ✅ Loaded model with {len(self.id2label)} labels")
        print(f"[BatchGPUNER] Labels: {list(self.id2label.values())[:10]}...")
    
    def batch_process(self, texts: list) -> list:
        """
        Process texts in batches on GPU.
        
        Args:
            texts: List of text strings
        
        Returns:
            List of entity lists, one per input text
        """
        from tqdm import tqdm
        
        all_results = []
        
        for i in tqdm(range(0, len(texts), self.batch_size), desc="GPU NER Batches"):
            batch_texts = texts[i:i + self.batch_size]
            batch_results = self._process_batch(batch_texts)
            all_results.extend(batch_results)
        
        return all_results
    
    def _process_batch(self, texts: list) -> list:
        """Process a single batch"""
        # Tokenize batch
        encoded = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
            return_offsets_mapping=True,
        )
        
        # Move to device
        input_ids = encoded["input_ids"].to(self.device)
        attention_mask = encoded["attention_mask"].to(self.device)
        offset_mapping = encoded["offset_mapping"]  # Keep on CPU for decoding
        
        # Run inference
        with torch.no_grad():
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            predictions = torch.argmax(outputs.logits, dim=-1)
            probabilities = torch.softmax(outputs.logits, dim=-1)
            max_probs = torch.max(probabilities, dim=-1).values
        
        # Decode results
        predictions = predictions.cpu().numpy()
        max_probs = max_probs.cpu().numpy()
        
        batch_results = []
        for idx, text in enumerate(texts):
            entities = self._decode_entities(
                text,
                predictions[idx],
                max_probs[idx],
                offset_mapping[idx],
            )
            batch_results.append(entities)
        
        return batch_results
    
    def _decode_entities(
        self, 
        text: str, 
        predictions: list, 
        probs: list,
        offset_mapping: list,
    ) -> list:
        """Decode BIO tags to entity spans"""
        entities = []
        current_entity = None
        current_start = None
        current_scores = []
        
        for idx, (pred_id, prob, offsets) in enumerate(zip(predictions, probs, offset_mapping)):
            # Skip special tokens
            if offsets[0] == offsets[1]:
                continue
            
            label = self.id2label.get(pred_id, "O")
            
            if label.startswith("B-"):
                # Save previous entity if exists
                if current_entity is not None:
                    avg_score = sum(current_scores) / len(current_scores) if current_scores else 0
                    if avg_score >= self.score_threshold:
                        entity_text = text[current_start:offsets[0]].strip()
                        if len(entity_text) > 2:
                            entities.append({
                                "text": entity_text.lower(),
                                "type": current_entity,
                                "score": avg_score,
                            })
                
                # Start new entity
                current_entity = label[2:]
                current_start = offsets[0].item() if hasattr(offsets[0], 'item') else offsets[0]
                current_scores = [prob]
                
            elif label.startswith("I-") and current_entity == label[2:]:
                # Continue current entity
                current_scores.append(prob)
                
            else:
                # End current entity
                if current_entity is not None:
                    avg_score = sum(current_scores) / len(current_scores) if current_scores else 0
                    if avg_score >= self.score_threshold:
                        end_offset = offsets[0].item() if hasattr(offsets[0], 'item') else offsets[0]
                        entity_text = text[current_start:end_offset].strip()
                        if len(entity_text) > 2:
                            entities.append({
                                "text": entity_text.lower(),
                                "type": current_entity,
                                "score": avg_score,
                            })
                    current_entity = None
                    current_start = None
                    current_scores = []
        
        # Handle last entity
        if current_entity is not None:
            avg_score = sum(current_scores) / len(current_scores) if current_scores else 0
            if avg_score >= self.score_threshold:
                entity_text = text[current_start:].strip()
                if len(entity_text) > 2:
                    entities.append({
                        "text": entity_text.lower(),
                        "type": current_entity,
                        "score": avg_score,
                    })
        
        return entities
    
    def batch_ner_with_sentences(
        self, 
        hash_id_to_text: dict,
        spacy_nlp=None,
    ) -> tuple:
        """
        Process passages with sentence-level entity extraction.
        Compatible with SpacyNER.batch_ner() interface.
        
        Args:
            hash_id_to_text: Dict mapping hash_id to passage text
            spacy_nlp: Optional spaCy model for sentence segmentation
        
        Returns:
            Tuple of (passage_hash_id_to_entities, sentence_to_entities)
        """
        import spacy
        from tqdm import tqdm
        
        # Load spaCy for sentence segmentation if not provided
        if spacy_nlp is None:
            try:
                spacy_nlp = spacy.load("en_core_web_sm", disable=["ner", "parser"])
                spacy_nlp.add_pipe("sentencizer")
            except:
                spacy_nlp = spacy.blank("en")
                spacy_nlp.add_pipe("sentencizer")
        
        passage_hash_id_to_entities = {}
        sentence_to_entities = {}
        
        # Collect all texts and sentences
        hash_ids = list(hash_id_to_text.keys())
        passages = list(hash_id_to_text.values())
        
        # Extract sentences
        all_sentences = []
        sentence_to_passage = {}  # sentence -> (hash_id, sent_idx)
        
        print("[BatchGPUNER] Extracting sentences...")
        for hash_id, passage in tqdm(hash_id_to_text.items(), desc="Sentence extraction"):
            doc = spacy_nlp(passage)
            for sent_idx, sent in enumerate(doc.sents):
                sent_text = sent.text.strip()
                if len(sent_text) >= 20:
                    all_sentences.append(sent_text)
                    sentence_to_passage[sent_text] = (hash_id, sent_idx)
        
        # Batch process passages
        print("[BatchGPUNER] Processing passages...")
        passage_results = self.batch_process(passages)
        
        for hash_id, entities in zip(hash_ids, passage_results):
            if entities:
                passage_hash_id_to_entities[hash_id] = set(e["text"] for e in entities)
        
        # Batch process sentences
        if all_sentences:
            print(f"[BatchGPUNER] Processing {len(all_sentences)} sentences...")
            sentence_results = self.batch_process(all_sentences)
            
            for sent_text, entities in zip(all_sentences, sentence_results):
                if entities:
                    sentence_to_entities[sent_text] = set(e["text"] for e in entities)
        
        print(f"[BatchGPUNER] ✅ Processed {len(passage_hash_id_to_entities)} passages with entities")
        print(f"[BatchGPUNER] ✅ Extracted {len(sentence_to_entities)} sentences with entities")
        
        return passage_hash_id_to_entities, sentence_to_entities


class CachedNER:
    """
    NER with multi-level caching support.
    Wraps SpacyNER or BatchGPUNER with cache layer.
    """
    
    def __init__(
        self,
        base_ner,
        cache_manager=None,
    ):
        """
        Initialize cached NER.
        
        Args:
            base_ner: Underlying NER model (SpacyNER or BatchGPUNER)
            cache_manager: Optional MultiLevelCache instance
        """
        self.base_ner = base_ner
        self.cache = cache_manager
    
    def batch_ner(self, hash_id_to_text: dict, max_workers: int = 8) -> tuple:
        """
        Batch NER with caching.
        
        Args:
            hash_id_to_text: Dict mapping hash_id to passage text
            max_workers: Number of parallel workers
        
        Returns:
            Tuple of (passage_hash_id_to_entities, sentence_to_entities)
        """
        from tqdm import tqdm
        import hashlib
        
        passage_hash_id_to_entities = {}
        sentence_to_entities = {}
        
        # Separate cached and uncached
        uncached_hash_id_to_text = {}
        
        if self.cache:
            print("[CachedNER] Checking cache...")
            for hash_id, text in tqdm(hash_id_to_text.items(), desc="Cache check"):
                doc_hash = hashlib.md5(text.encode()).hexdigest()
                cached = self.cache.get_ner(doc_hash)
                
                if cached:
                    # Use cached result
                    if "passage_entities" in cached:
                        passage_hash_id_to_entities[hash_id] = set(cached["passage_entities"])
                    if "sentence_to_entities" in cached:
                        for sent, ents in cached["sentence_to_entities"].items():
                            sentence_to_entities[sent] = set(ents)
                else:
                    uncached_hash_id_to_text[hash_id] = text
            
            print(f"[CachedNER] Cache hits: {len(hash_id_to_text) - len(uncached_hash_id_to_text)}")
            print(f"[CachedNER] Cache misses: {len(uncached_hash_id_to_text)}")
        else:
            uncached_hash_id_to_text = hash_id_to_text
        
        # Process uncached
        if uncached_hash_id_to_text:
            if hasattr(self.base_ner, 'batch_ner_with_sentences'):
                new_passage_entities, new_sentence_entities = self.base_ner.batch_ner_with_sentences(
                    uncached_hash_id_to_text
                )
            else:
                new_passage_entities, new_sentence_entities = self.base_ner.batch_ner(
                    uncached_hash_id_to_text, max_workers
                )
            
            # Merge results
            passage_hash_id_to_entities.update(new_passage_entities)
            sentence_to_entities.update(new_sentence_entities)
            
            # Update cache
            if self.cache:
                print("[CachedNER] Updating cache...")
                for hash_id, text in uncached_hash_id_to_text.items():
                    doc_hash = hashlib.md5(text.encode()).hexdigest()
                    cache_entry = {
                        "passage_entities": list(new_passage_entities.get(hash_id, set())),
                        "sentence_to_entities": {
                            k: list(v) for k, v in new_sentence_entities.items()
                            if k in text  # Only cache sentences from this passage
                        }
                    }
                    self.cache.set_ner(doc_hash, cache_entry)
                
                # Persist cache
                self.cache.ner_cache.save()
        
        return passage_hash_id_to_entities, sentence_to_entities