#!/usr/bin/env python3
"""
HyperLinearRAG Evaluation Script

Evaluates the HyperLinearRAG system on 5 medical QA benchmarks:
- MedQA
- MedMCQA
- MMLU (medical subset)
- PubMedQA
- BioASQ

Usage:
    python scripts/evaluate_hyperlinearrag.py --datasets medqa medmcqa mmlu pubmedqa bioasq
    python scripts/evaluate_hyperlinearrag.py --baseline  # Run baseline LinearRAG for comparison
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime
from typing import Dict, List, Any

import torch
from sentence_transformers import SentenceTransformer

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import LinearRAGConfig
from src.LinearRAG import LinearRAG
from src.llm import LLM
from src.dataset_loader import load_mirage_questions_local, load_pubmed_passages
from src.eval.summary import summarize_results


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate HyperLinearRAG on medical QA benchmarks")
    
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["medqa", "medmcqa", "mmlu", "pubmedqa", "bioasq"],
        help="Datasets to evaluate on"
    )
    parser.add_argument(
        "--baseline",
        action="store_true",
        help="Run baseline LinearRAG (without hypergraph)"
    )
    parser.add_argument(
        "--questions_limit",
        type=int,
        default=None,
        help="Limit number of questions per dataset (for quick testing)"
    )
    parser.add_argument(
        "--chunks_limit",
        type=int,
        default=None,
        help="Limit number of passages/chunks"
    )
    parser.add_argument(
        "--embedding_model",
        type=str,
        default="BAAI/bge-base-en-v1.5",
        help="Embedding model to use"
    )
    parser.add_argument(
        "--llm_model",
        type=str,
        default="qwen2.5:7b-instruct",
        help="LLM model for answer generation"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="experiments/hyperlinearrag",
        help="Output directory for results"
    )
    parser.add_argument(
        "--reuse_index",
        type=str,
        default=None,
        help="Reuse existing index from specified namespace"
    )
    parser.add_argument(
        "--max_workers",
        type=int,
        default=4,
        help="Max parallel workers"
    )
    
    # HyperLinearRAG specific parameters
    parser.add_argument(
        "--hyperedge_top_k",
        type=int,
        default=30,
        help="Top-k hyperedges to retrieve"
    )
    parser.add_argument(
        "--hyperedge_threshold",
        type=float,
        default=0.3,
        help="Hyperedge retrieval threshold"
    )
    parser.add_argument(
        "--hyperedge_node_weight",
        type=float,
        default=1.2,
        help="Weight for hyperedge nodes in PPR"
    )
    
    return parser.parse_args()


def load_data(args) -> tuple:
    """Load questions and passages"""
    print(f"\n{'='*60}")
    print(f"Loading data...")
    print(f"{'='*60}")
    
    # 加载问题（每个数据集限制 questions_limit 个）
    questions = load_mirage_questions_local(args.datasets, args.questions_limit)
    
    # 加载 passages
    passages = load_pubmed_passages(
        chunks_dir="dataset/pubmed/chunk",
        chunks_limit=args.chunks_limit
    )
    
    print(f"Loaded {len(questions)} questions from {args.datasets}")
    
    # 按数据集统计
    dataset_counts = {}
    for q in questions:
        ds = q.get("dataset", "unknown")
        dataset_counts[ds] = dataset_counts.get(ds, 0) + 1
    for ds, count in dataset_counts.items():
        print(f"  - {ds}: {count} questions")
    
    print(f"Loaded {len(passages)} passages")
    
    return questions, passages


def create_config(args, embedding_model, llm_model) -> LinearRAGConfig:
    """Create LinearRAGConfig with HyperLinearRAG parameters"""
    
    # Determine index namespace
    if args.reuse_index:
        index_namespace = args.reuse_index
    else:
        index_namespace = f"hyperlinearrag_{'_'.join(args.datasets)}"
    
    config = LinearRAGConfig(
        embedding_model=embedding_model,
        dataset_name=index_namespace,
        spacy_model="en_ner_bc5cdr_md",
        max_workers=args.max_workers,
        llm_model=llm_model,
        use_hf_ner=True,
        use_enhanced_ner=True,
        working_dir="import",
        # HyperLinearRAG specific
        use_hypergraph=not args.baseline,
        hyperedge_top_k=args.hyperedge_top_k,
        hyperedge_retrieval_threshold=args.hyperedge_threshold,
        hyperedge_node_weight=args.hyperedge_node_weight,
        enable_incremental_index=True,
        enable_multi_level_cache=True,
    )
    
    return config


def run_evaluation(args, questions, passages, config) -> Dict[str, Any]:
    """Run evaluation and return results"""
    
    print(f"\n{'='*60}")
    print(f"Running {'Baseline LinearRAG' if args.baseline else 'HyperLinearRAG'}")
    print(f"{'='*60}")
    
    # Initialize model
    rag_model = LinearRAG(global_config=config)
    
    # Index passages
    start_time = time.time()
    rag_model.index(passages)
    index_time = time.time() - start_time
    print(f"Indexing completed in {index_time:.2f}s")
    
    # Run QA
    start_time = time.time()
    results = rag_model.qa(questions)
    qa_time = time.time() - start_time
    print(f"QA completed in {qa_time:.2f}s")
    
    # Summarize results
    summary = summarize_results(results)
    
    # Add timing info
    summary["index_time_seconds"] = index_time
    summary["qa_time_seconds"] = qa_time
    summary["total_time_seconds"] = index_time + qa_time
    summary["is_baseline"] = args.baseline
    summary["hypergraph_enabled"] = not args.baseline
    
    # Add hypergraph stats if available
    if hasattr(rag_model, 'hypergraph_store') and rag_model.hypergraph_store:
        stats = rag_model.hypergraph_store.get_stats()
        summary["hypergraph_stats"] = {
            "num_hyperedges": stats.num_hyperedges,
            "num_entities": stats.num_entities,
            "avg_entities_per_hyperedge": stats.avg_entities_per_hyperedge,
        }
    
    return results, summary


def save_results(args, results, summary):
    """Save results to files"""
    os.makedirs(args.output_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    mode = "baseline" if args.baseline else "hypergraph"
    datasets_str = "_".join(args.datasets)
    
    results_path = os.path.join(args.output_dir, f"results_{mode}_{datasets_str}_{timestamp}.json")
    summary_path = os.path.join(args.output_dir, f"summary_{mode}_{datasets_str}_{timestamp}.json")
    
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    
    print(f"\nResults saved to:")
    print(f"  {results_path}")
    print(f"  {summary_path}")
    
    return results_path, summary_path


def print_summary(summary: Dict[str, Any]):
    """Print evaluation summary"""
    print(f"\n{'='*60}")
    print(f"EVALUATION SUMMARY")
    print(f"{'='*60}")
    
    print(f"\nMode: {'Baseline LinearRAG' if summary.get('is_baseline') else 'HyperLinearRAG'}")
    
    print(f"\nAccuracy by Dataset:")
    for dataset, metrics in summary.get("by_dataset", {}).items():
        acc = metrics.get("accuracy", 0)
        total = metrics.get("total", 0)
        correct = metrics.get("correct", 0)
        print(f"  {dataset}: {acc:.4f} ({correct}/{total})")
    
    overall_acc = summary.get("overall_accuracy", 0)
    print(f"\nOverall Accuracy: {overall_acc:.4f}")
    
    print(f"\nTiming:")
    print(f"  Index time: {summary.get('index_time_seconds', 0):.2f}s")
    print(f"  QA time: {summary.get('qa_time_seconds', 0):.2f}s")
    print(f"  Total time: {summary.get('total_time_seconds', 0):.2f}s")
    
    if "hypergraph_stats" in summary:
        stats = summary["hypergraph_stats"]
        print(f"\nHypergraph Statistics:")
        print(f"  Hyperedges: {stats.get('num_hyperedges', 0)}")
        print(f"  Entities: {stats.get('num_entities', 0)}")
        print(f"  Avg entities/hyperedge: {stats.get('avg_entities_per_hyperedge', 0):.2f}")
    
    print(f"{'='*60}\n")


def main():
    args = parse_args()
    
    print(f"\n{'='*60}")
    print(f"HyperLinearRAG Evaluation")
    print(f"{'='*60}")
    print(f"Datasets: {args.datasets}")
    print(f"Mode: {'Baseline' if args.baseline else 'HyperLinearRAG'}")
    print(f"Output: {args.output_dir}")
    
    # Load embedding model
    print(f"\nLoading embedding model: {args.embedding_model}")
    embedding_model = SentenceTransformer(args.embedding_model)
    embedding_model.eval()
    
    # Load LLM
    print(f"Loading LLM: {args.llm_model}")
    llm_model = LLM(model_name=args.llm_model)
    
    # Load data
    questions, passages = load_data(args)
    
    if not questions:
        print("ERROR: No questions loaded!")
        return
    if not passages:
        print("ERROR: No passages loaded!")
        return
    
    # Create config
    config = create_config(args, embedding_model, llm_model)
    
    # Run evaluation
    results, summary = run_evaluation(args, questions, passages, config)
    
    # Save results
    save_results(args, results, summary)
    
    # Print summary
    print_summary(summary)


if __name__ == "__main__":
    main()

