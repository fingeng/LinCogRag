#!/usr/bin/env python3
"""
LinCog-RAG Benchmark Runner

Note: Run with PYTHONUNBUFFERED=1 for real-time output

This script runs the LinearRAG + HyperGraph pipeline on 5 MIRAGE datasets:
- medqa, medmcqa, mmlu, pubmedqa, bioasq

Configuration:
- NER: models/biomedical-ner-all (HuggingFace biomedical NER)
- Chunks: 20000 (prioritizing pubmedqa-related papers)
- LLM: GPT-4o
"""

import os
import sys
import json
import glob
import time
from datetime import datetime
from typing import List, Dict, Any, Tuple

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

# ===================== Configuration =====================
CHUNKS_LIMIT = 20000
QUESTIONS_LIMIT = None  # None = 使用全部问题
LLM_MODEL = "gpt-4o"
EMBEDDING_MODEL = "model/all-mpnet-base-v2"  # Use local model to avoid network issues
SPACY_MODEL = "en_ner_bc5cdr_md"  # BC5CDR as primary
MAX_WORKERS = 8

# Datasets to evaluate
MIRAGE_DATASETS = ["medqa", "medmcqa", "mmlu", "pubmedqa", "bioasq"]

# Output directory
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "artifacts", "lincog_benchmark")
INDEX_NAMESPACE = "lincog_20k_pubmedqa"


def load_pubmed_passages_prioritized(
    chunks_dir: str = "dataset/pubmed/chunk",
    pubmedqa_priority_file: str = "pubmed_with_pubmedqa.jsonl",
    chunks_limit: int = 20000,
) -> List[str]:
    """
    Load PubMed passages with priority to pubmedqa-related papers.
    
    Strategy:
    1. First load all pubmedqa-related papers (500)
    2. Then fill up to chunks_limit with regular pubmed papers
    """
    passages: List[str] = []
    
    chunks_dir = os.path.join(PROJECT_ROOT, chunks_dir)
    if not os.path.exists(chunks_dir):
        raise FileNotFoundError(f"PubMed chunks dir not found: {chunks_dir}")
    
    # Step 1: Load pubmedqa-related papers first
    priority_path = os.path.join(chunks_dir, pubmedqa_priority_file)
    if os.path.exists(priority_path):
        print(f"[DataLoader] Loading priority file: {priority_path}")
        with open(priority_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    item = json.loads(line)
                    text = item.get("text") or item.get("contents") or item.get("content") or ""
                    text = text.strip()
                    if text:
                        passages.append(text)
                except json.JSONDecodeError:
                    continue
        print(f"[DataLoader] Loaded {len(passages)} pubmedqa-related passages")
    
    # Step 2: Fill with regular pubmed papers
    remaining = chunks_limit - len(passages)
    if remaining > 0:
        jsonl_files = sorted(glob.glob(os.path.join(chunks_dir, "pubmed23n*.jsonl")))
        print(f"[DataLoader] Loading {remaining} more passages from {len(jsonl_files)} files...")
        
        for file_path in jsonl_files:
            if len(passages) >= chunks_limit:
                break
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    for line in f:
                        if len(passages) >= chunks_limit:
                            break
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            item = json.loads(line)
                            text = item.get("contents") or item.get("text") or item.get("content") or ""
                            text = text.strip()
                            if text:
                                passages.append(text)
                        except json.JSONDecodeError:
                            continue
            except Exception as e:
                print(f"[DataLoader] Error reading {file_path}: {e}")
                continue
    
    print(f"[DataLoader] Total passages loaded: {len(passages)}")
    return passages


def run_benchmark():
    """Run the full benchmark pipeline."""
    import torch
    from sentence_transformers import SentenceTransformer
    
    from src.config import LinearRAGConfig
    from src.dataset_loader import load_mirage_questions_local
    from src.LinearRAG import LinearRAG
    from src.llm import LLM
    from src.eval.summary import summarize_results
    
    # Setup output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"lincog_5datasets_{timestamp}"
    
    print("\n" + "="*70)
    print("LinCog-RAG Benchmark Runner")
    print("="*70)
    print(f"Timestamp: {timestamp}")
    print(f"Chunks limit: {CHUNKS_LIMIT}")
    print(f"LLM Model: {LLM_MODEL}")
    print(f"Datasets: {MIRAGE_DATASETS}")
    print("="*70 + "\n")
    
    # Step 1: Load embedding model
    print("[Step 1] Loading embedding model...")
    embedding_model = SentenceTransformer(EMBEDDING_MODEL)
    embedding_model.eval()
    if torch.cuda.is_available():
        embedding_model = embedding_model.cuda()
        print(f"[Step 1] ✅ Loaded on CUDA")
    else:
        print(f"[Step 1] ✅ Loaded on CPU")
    
    # Step 2: Load passages
    print("\n[Step 2] Loading PubMed passages...")
    passages = load_pubmed_passages_prioritized(chunks_limit=CHUNKS_LIMIT)
    print(f"[Step 2] ✅ Loaded {len(passages)} passages")
    
    # Step 3: Load questions from all datasets
    print("\n[Step 3] Loading questions from MIRAGE datasets...")
    all_questions = load_mirage_questions_local(
        MIRAGE_DATASETS, 
        questions_limit=QUESTIONS_LIMIT,
        mirage_root=os.path.join(PROJECT_ROOT, "MIRAGE", "rawdata")
    )
    print(f"[Step 3] ✅ Loaded {len(all_questions)} questions total")
    
    # Count by dataset
    dataset_counts = {}
    for q in all_questions:
        ds = q.get("dataset", "unknown")
        dataset_counts[ds] = dataset_counts.get(ds, 0) + 1
    for ds, count in sorted(dataset_counts.items()):
        print(f"         - {ds}: {count} questions")
    
    # Step 4: Initialize LLM
    print("\n[Step 4] Initializing LLM...")
    llm_model = LLM(model_name=LLM_MODEL)
    print(f"[Step 4] ✅ LLM initialized: {LLM_MODEL}")
    
    # Step 5: Initialize LinearRAG with HyperGraph
    print("\n[Step 5] Initializing LinearRAG with HyperGraph...")
    config = LinearRAGConfig(
        embedding_model=embedding_model,
        dataset_name=INDEX_NAMESPACE,
        spacy_model=SPACY_MODEL,
        max_workers=MAX_WORKERS,
        llm_model=llm_model,
        use_hf_ner=True,  # Enable HuggingFace NER (biomedical-ner-all)
        use_enhanced_ner=True,
        working_dir=os.path.join(PROJECT_ROOT, "import"),
        # HyperGraph settings
        use_hypergraph=True,
        min_entities_per_hyperedge=2,
        max_hyperedge_score_boost=1.5,
        hyperedge_top_k=30,
        hyperedge_retrieval_threshold=0.3,
        # Retrieval settings
        retrieval_top_k=5,
        candidate_pool_size=500,
        use_candidate_filtering=True,
    )
    
    rag_model = LinearRAG(global_config=config)
    print(f"[Step 5] ✅ LinearRAG initialized")
    
    # Step 6: Build index
    print("\n[Step 6] Building index (NER + Graph + HyperGraph)...")
    index_start = time.time()
    rag_model.index(passages)
    index_time = time.time() - index_start
    print(f"[Step 6] ✅ Index built in {index_time:.2f}s")
    
    # Step 7: Run QA
    print("\n[Step 7] Running QA on all questions...")
    qa_start = time.time()
    results = rag_model.qa(all_questions)
    qa_time = time.time() - qa_start
    print(f"[Step 7] ✅ QA completed in {qa_time:.2f}s")
    
    # Step 8: Summarize results
    print("\n[Step 8] Summarizing results...")
    summary = summarize_results(results)
    
    # Add timing info
    summary["index_time_seconds"] = index_time
    summary["qa_time_seconds"] = qa_time
    summary["total_time_seconds"] = index_time + qa_time
    summary["config"] = {
        "chunks_limit": CHUNKS_LIMIT,
        "llm_model": LLM_MODEL,
        "embedding_model": EMBEDDING_MODEL,
        "datasets": MIRAGE_DATASETS,
        "use_hypergraph": True,
    }
    
    # Save results
    results_path = os.path.join(OUTPUT_DIR, f"{run_name}_results.json")
    summary_path = os.path.join(OUTPUT_DIR, f"{run_name}_summary.json")
    
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    
    # Print final summary
    print("\n" + "="*70)
    print("📊 BENCHMARK RESULTS")
    print("="*70)
    print(f"Total questions:         {summary['total_questions']}")
    print(f"LLM Accuracy:            {summary['overall_llm_accuracy']:.2f}%")
    print(f"Contain Accuracy:        {summary['overall_contain_accuracy']:.2f}%")
    print(f"Questions w/o entities:  {summary['questions_wo_entities']}")
    print(f"Invalid answers:         {summary['total_invalid']}")
    print(f"Valid answer rate:       {summary['valid_answer_rate']:.2f}%")
    print("-"*70)
    print("Per-dataset breakdown:")
    
    if "dataset_stats" in summary:
        for ds, stats in summary["dataset_stats"].items():
            total = stats.get("total", 0)
            correct = stats.get("correct", 0)
            acc = (correct / total * 100) if total > 0 else 0
            print(f"  {ds:15} {acc:6.2f}% ({correct}/{total})")
    
    print("-"*70)
    print(f"Index time:  {index_time:.2f}s")
    print(f"QA time:     {qa_time:.2f}s")
    print(f"Total time:  {index_time + qa_time:.2f}s")
    print("="*70)
    print(f"\n💾 Results saved to: {results_path}")
    print(f"💾 Summary saved to: {summary_path}\n")
    
    return results, summary


if __name__ == "__main__":
    run_benchmark()

