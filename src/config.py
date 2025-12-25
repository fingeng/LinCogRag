class LinearRAGConfig:
    def __init__(
        self,
        embedding_model,
        dataset_name,
        spacy_model="en_ner_bc5cdr_md",  # 🔧 Default to BC5CDR
        max_workers=4,  # ✅ 限制并发，避免SSH断连
        llm_model=None,
        use_hf_ner=True,  # 🔧 ADD: Enable HF NER by default
        use_enhanced_ner=True,  # 🔧 ADD: Enable enhanced NER by default
        working_dir="import",  # 🔧 ADD: Default working directory
        batch_size=32,
        retrieval_top_k=3,  # 🔧 降低噪声: 5→3
        max_iterations=2,  # ✅ 优化: 3→2
        iteration_threshold=0.3,  # ✅ 优化: 0.1→0.3
        top_k_sentence=5,  # ✅ 优化: 3→5
        # ✅ 新增: 候选集预筛选参数
        use_candidate_filtering=True,  # 启用候选集过滤
        candidate_pool_size=500,  # ✅ 优化: 200→500 提高准确率
        passage_ratio=0.7,
        passage_node_weight=1.0,
        damping=0.85,
        # ==================== HyperLinearRAG Parameters ====================
        # Hypergraph construction
        use_hypergraph=True,  # Enable hypergraph mode
        min_entities_per_hyperedge=2,  # Min entities to form a hyperedge
        max_hyperedge_score_boost=1.5,  # Max score boost for medical patterns
        # GPU batch processing
        ner_batch_size=32,  # NER batch size for GPU processing
        embedding_batch_size=64,  # Embedding batch size
        use_gpu_ner=True,  # Enable GPU batch NER
        # Incremental indexing
        enable_incremental_index=True,  # Enable incremental updates
        # Caching
        enable_multi_level_cache=True,  # Enable multi-level caching
        cache_dir="cache",  # Cache directory
        # Hypergraph retrieval
        hyperedge_top_k=30,  # Top-k hyperedges to retrieve
        hyperedge_node_weight=1.2,  # Weight for hyperedge nodes in PPR
        hyperedge_retrieval_threshold=0.3,  # Threshold for hyperedge retrieval
        hyperedge_entity_boost=1.2,  # Boost for passages with expanded entities
    ):
        # Model parameters
        self.embedding_model = embedding_model
        self.spacy_model = spacy_model
        self.llm_model = llm_model
        
        # NER strategy
        self.use_hf_ner = use_hf_ner
        self.use_enhanced_ner = use_enhanced_ner
        
        # Dataset parameters
        self.dataset_name = dataset_name
        self.working_dir = working_dir
        
        # Processing parameters
        self.max_workers = max_workers
        self.batch_size = batch_size
        
        # Retrieval parameters
        self.retrieval_top_k = retrieval_top_k
        self.max_iterations = max_iterations
        self.iteration_threshold = iteration_threshold
        self.top_k_sentence = top_k_sentence
        
        # ✅ 候选集预筛选参数
        self.use_candidate_filtering = use_candidate_filtering
        self.candidate_pool_size = candidate_pool_size
        
        # Graph parameters
        self.passage_ratio = passage_ratio
        self.passage_node_weight = passage_node_weight
        self.damping = damping
        
        # ==================== HyperLinearRAG Parameters ====================
        # Hypergraph construction
        self.use_hypergraph = use_hypergraph
        self.min_entities_per_hyperedge = min_entities_per_hyperedge
        self.max_hyperedge_score_boost = max_hyperedge_score_boost
        
        # GPU batch processing
        self.ner_batch_size = ner_batch_size
        self.embedding_batch_size = embedding_batch_size
        self.use_gpu_ner = use_gpu_ner
        
        # Incremental indexing
        self.enable_incremental_index = enable_incremental_index
        
        # Caching
        self.enable_multi_level_cache = enable_multi_level_cache
        self.cache_dir = cache_dir
        
        # Hypergraph retrieval
        self.hyperedge_top_k = hyperedge_top_k
        self.hyperedge_node_weight = hyperedge_node_weight
        self.hyperedge_retrieval_threshold = hyperedge_retrieval_threshold
        self.hyperedge_entity_boost = hyperedge_entity_boost