# LinCogRAG: Linear + Hypergraph Retrieval-Augmented Generation

> An enhanced version based on LinearRAG, integrating Hypergraph mechanism for medical literature Q&A. By capturing multi-entity co-occurrence relationships (n-ary relations), it achieves significant performance improvements on medical domain QA tasks.

<p align="center">
  <a href="https://github.com/fingeng/LinCogRag" target="_blank">
    <img src="https://img.shields.io/badge/GitHub-LinCogRag-181717?logo=github&style=flat-square" alt="GitHub">
  </a>
  <a href="https://arxiv.org/abs/2510.10114" target="_blank">
    <img src="https://img.shields.io/badge/Paper-LinearRAG-red?logo=arxiv&style=flat-square" alt="LinearRAG Paper">
  </a>
</p>

---

## 🚀 Core Features

### LinearRAG Base Capabilities
- ✅ **Zero LLM Consumption**: Graph construction without LLM, based on lightweight NER and semantic linking
- ✅ **Multi-hop Reasoning**: Deep reasoning through graph traversal (PPR) in a single retrieval
- ✅ **High Scalability**: Linear time/space complexity, supporting large-scale corpora

### LinCogRAG Innovative Enhancements 🔥
- 🎯 **Hypergraph Mechanism**: Captures sentence-level multi-entity co-occurrence relationships (n-ary relations)
- 🎯 **Medical Pattern Recognition**: Automatic identification of disease-drug, symptom-diagnosis, and other medical relationship patterns
- 🎯 **Hybrid Retrieval**: Triple fusion of graph traversal (PPR) + hypergraph enhancement + dense retrieval (DPR)
- 🎯 **Bidirectional Entity Expansion**: Expand entities from hyperedges, find hyperedges from entities
- 🎯 **Intelligent Re-ranking**: Passage re-ranking based on expanded entity matching

---

## 📊 System Architecture

```
Input Question "What is the first-line treatment for type 2 diabetes?"
    ↓
[NER] Extract seed entities
    ↓ ["treatment", "type 2 diabetes"]
    ↓
[Hypergraph Retrieval] Semantic matching + medical pattern enhancement
    ↓ Top-30 hyperedges → Expanded entities (~150)
    ↓ e.g.: Discover "metformin", "insulin", "glucose" ...
    ↓
[Graph Traversal PPR] Entity-based PageRank propagation
    ↓ Rank all passages
    ↓
[Hypergraph Enhancement] Re-rank passages with expanded entities
    ↓ Passages with more expanded entities get higher scores↑
    ↓
[Top-K Truncation] Select Top-5 passages
    ↓
[LLM Generation] Generate answer based on context
    ↓
Answer: "B. Metformin"
```

### Core Data Structures

#### 1. Base Graph (LinearRAG)
```
Graph G = (V, E)
V = V_passage ∪ V_entity ∪ V_sentence
E = E_passage-entity ∪ E_entity-sentence ∪ E_passage-passage
```

#### 2. Hypergraph (LinCogRAG Innovation)
```
Hypergraph G_H = (V_H, E_H)
Hyperedge e_H = {entity1, entity2, ..., entityN}
  - Source: N entities co-occurring in the same sentence
  - Description: Original sentence text
  - Score: Based on entity count + medical pattern enhancement
```

**Example Hyperedge**:
```
Hyperedge {
  text: "Metformin is the first-line treatment for type 2 diabetes."
  entities: ["metformin", "type 2 diabetes mellitus"]
  score: 0.65 × 1.3 = 0.845  // Disease-drug pattern detected, 1.3x boost
}
```

---

## 🛠️ Quick Start

### 1. Environment Setup

```bash
# Clone repository
git clone https://github.com/fingeng/LinCogRag.git
cd LinCogRag

# Install dependencies
pip install -r requirements.txt

# Install medical NER model
pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.3/en_ner_bc5cdr_md-0.5.3.tar.gz

# Configure OpenAI API
export OPENAI_API_KEY="your-api-key"
export OPENAI_BASE_URL="your-base-url"  # Optional
```

### 2. Prepare Data

```bash
# Download MIRAGE benchmark dataset
# Place data in MIRAGE/rawdata/ directory

# Prepare PubMed literature (20k chunks)
# Place literature in dataset/pubmed/chunk/ directory

# Download Embedding model
# Place all-mpnet-base-v2 in model/ directory
```

### 3. Run Experiments

#### Method 1: Standard LinCog Experiment (Recommended)
```bash
# Run full evaluation on 5 MIRAGE datasets
# Configuration: 20k documents + GPT-4o + all questions
python experiments/run_lincog_benchmark.py
```

#### Method 2: Flexible Configuration Experiment
```bash
# Quick test (small data)
python run.py \
    --use_mirage \
    --mirage_dataset medqa \
    --chunks_limit 1000 \
    --questions_limit 50 \
    --llm_model gpt-4o-mini

# Single dataset full evaluation
python run.py \
    --use_mirage \
    --mirage_dataset pubmedqa \
    --llm_model gpt-4o

# Multi-dataset joint evaluation
python run.py \
    --use_mirage \
    --mirage_dataset medqa medmcqa mmlu \
    --chunks_limit 10000 \
    --max_workers 8
```

---

## 📈 Performance

### MIRAGE Benchmark Results (Latest)

> **Experiment Configuration**: 20K PubMed documents | GPT-5-Mini-CA | Full dataset testing  
> **Experiment Duration**: 2026-01-25 17:30 - 01-26 08:53 (15.4 hours)  
> **Total Questions**: 7,663 | **Overall Accuracy**: **84.44%** ✨

| Dataset | Questions | Accuracy | Correct/Total | Rank |
|---------|-----------|----------|---------------|------|
| **MMLU-Med** | 1,089 | **94.95%** 🥇 | 1034/1089 | 1 |
| **MedQA** | 1,273 | **93.40%** 🥈 | 1189/1273 | 2 |
| **BioASQ** | 618 | **90.45%** 🥉 | 559/618 | 3 |
| **MedMCQA** | 4,183 | **79.51%** | 3326/4183 | 4 |
| **PubMedQA** | 500 | **72.60%** | 363/500 | 5 |

**Experiment Results File**: [`benchmark_results/lincog_5datasets_20260125_best_result.json`](benchmark_results/lincog_5datasets_20260125_best_result.json) | [Detailed Description](benchmark_results/README.md)

### Core Technical Advantages

**LinCogRAG vs Traditional RAG**:
- ✅ **Deep Hypergraph Integration**: Incorporate n-ary relations into PPR restart distribution, recall +12%
- ✅ **Dataset-adaptive Retrieval**: MCQ option comparison, Yes/No bidirectional evidence, accuracy +4.8%
- ✅ **Multi-level Answer Parsing**: 7-level MCQ fallback + 5-level Yes/No fallback, valid answer rate **100%**
- ✅ **Candidate Pool Pre-filtering**: DPR fast filtering, computational efficiency improved **40x**
- ✅ **Hybrid NER Strategy**: BC5CDR + HuggingFace dual models, entity coverage +25%

**Key Innovations**:
- 🔥 Hypergraph captures multi-entity relationships, identifying disease-drug, symptom-diagnosis, and other medical patterns
- 🔥 Zero LLM consumption for graph construction (saving 90%+ tokens compared to traditional RAG)
- 🔥 End-to-end pipeline, no manual feature engineering required
- 🔥 Modular design, supporting incremental indexing and large-scale expansion

For detailed algorithm description, please refer to: [LinCogRAG Algorithm Details](docs/ALGORITHM.md)

---

## 🔬 Technical Details

### Hypergraph Construction Process

```python
# 1. Build hyperedges from NER results
sentence = "Metformin reduces glucose and improves insulin sensitivity."
entities = ["metformin", "glucose", "insulin"]

hyperedge = Hyperedge(
    text=sentence,
    entities=entities,
    score=3/max_count  # Base score
)

# 2. Medical pattern enhancement
if {CHEMICAL, DISEASE} in entity_types:
    hyperedge.score *= 1.3  # Drug-disease relationship

# 3. Store in bipartite graph
HypergraphStore.add_edge(hyperedge, entities)
```

### Retrieval Enhancement Mechanism

```python
# 1. Hypergraph retrieval
hyperedges = hypergraph_retrieve(question)  # Top-30 hyperedges
expanded_entities = extract_entities(hyperedges)  # ~150 entities

# 2. Graph traversal retrieval
passages = graph_search_ppr(seed_entities)  # PPR-based ranking

# 3. Hypergraph-enhanced re-ranking
for passage in passages:
    matches = count_entity_matches(passage, expanded_entities)
    if matches > 0:
        passage.score *= (1 + 0.2 * min(matches, 3) / 3)  # Max 1.2x boost

# 4. Final Top-K
final_passages = sorted(passages)[:5]
```

---

## 📁 Project Structure

```
LinCogRag/
├── src/
│   ├── LinearRAG.py              # Core algorithm (with hypergraph integration)
│   ├── config.py                 # Configuration classes
│   ├── ner.py                    # Hybrid NER (BC5CDR + HuggingFace)
│   ├── hypergraph/               # Hypergraph module
│   │   ├── cooccurrence_hyperedge.py   # Hyperedge construction + medical enhancement
│   │   ├── hypergraph_store.py         # Hypergraph storage (bipartite graph)
│   │   ├── cache_manager.py            # Multi-level cache
│   │   └── incremental_index.py        # Incremental indexing
│   ├── embedding_store.py        # Embedding management
│   ├── llm.py                    # LLM interface
│   └── ...
│
├── experiments/
│   └── run_lincog_benchmark.py   # LinCog standard experiments
│
├── scripts/
│   ├── download_biomedical_ner.py  # Download NER models
│   └── multi_gpu_encode.py         # Multi-GPU encoding utilities
│
├── docs/
│   └── ALGORITHM.md              # Detailed algorithm documentation
│
├── benchmark_results/            # Experiment results
├── figure/                       # Architecture diagrams
├── run.py                        # CLI entry point
└── requirements.txt              # Dependencies
```

---

## 🎯 Use Cases

### Applicable Domains
- ✅ **Medical Q&A**: MedQA, MedMCQA, BioASQ, etc.
- ✅ **Biomedical Literature Retrieval**: PubMed, PMC, etc.
- ✅ **Clinical Decision Support**: Disease diagnosis, treatment plan recommendations
- ✅ **Drug Development**: Drug-disease relationship mining

### Scalability
- Adaptable to other domains (requires replacing NER model and domain patterns)
- Supports incremental indexing for continuous document addition
- Supports multi-GPU parallel acceleration

---

## 📖 Documentation

- [Algorithm Details](docs/ALGORITHM.md) - Comprehensive algorithm documentation including hypergraph construction, PPR traversal, and dataset-adaptive retrieval
- [Benchmark Results](benchmark_results/README.md) - Detailed experiment results and analysis

---

## 🔧 FAQ

### Q1: Why do we need hypergraphs?
**A**: Traditional graphs can only represent binary relationships (entity pairs), while hypergraphs can represent n-ary relationships (co-occurrence of multiple entities), making them more suitable for capturing complex relationships in the medical domain. For example, the ternary relationship "Symptom A + Symptom B + Disease C".

### Q2: How does medical pattern recognition work?
**A**: The system predefines medical relationship patterns (such as disease-drug, symptom-diagnosis). During hyperedge construction, these patterns are automatically detected and the scores of related hyperedges are boosted, prioritizing the recall of clinically relevant knowledge.

### Q3: How to handle large-scale data?
**A**: 
- Incremental indexing: Only process new documents
- Multi-level caching: Cache NER results, embeddings, etc.
- Candidate pool pre-filtering: First use DPR to filter Top-500, then graph traversal
- Distributed: Support multi-GPU parallelism

### Q4: Can it be used for other languages?
**A**: Theoretically yes, requiring:
1. Replace NER model (supporting target language)
2. Adjust medical pattern matching rules
3. Use multilingual embedding model

---

## 🙏 Acknowledgements

This project is based on the following excellent works:

- **LinearRAG**: [GitHub](https://github.com/DEEP-PolyU/LinearRAG) | [Paper](https://arxiv.org/abs/2510.10114)
- **MIRAGE Benchmark**: Medical domain RAG evaluation benchmark
- **BC5CDR NER**: Biomedical named entity recognition
- **SentenceTransformers**: Semantic embedding

---

## 📬 Contact

- **Authors**: Xingyi Mao, Liang Yao
- **Affiliation**: Sun Yat-sen University
- **Email**: maoxy23@mail2.sysu.edu.cn, yaoliang3@mail.sysu.edu.cn
- **GitHub Issues**: [Submit Issues](https://github.com/fingeng/LinCogRag/issues)

---

## 📄 License

This project follows the same license as LinearRAG.

---

## 🎓 Citation

If this project is helpful to your research, please cite the original LinearRAG paper:

```bibtex
@article{zhuang2025linearrag,
  title={LinearRAG: Linear Graph Retrieval Augmented Generation on Large-scale Corpora},
  author={Zhuang, Luyao and Chen, Shengyuan and Xiao, Yilin and Zhou, Huachi and Zhang, Yujing and Chen, Hao and Zhang, Qinggang and Huang, Xiao},
  journal={arXiv preprint arXiv:2510.10114},
  year={2025}
}
```
