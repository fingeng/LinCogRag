from __future__ import annotations

import json
import os
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import torch
from sentence_transformers import SentenceTransformer

from src.config import LinearRAGConfig
from src.dataset_loader import load_mirage_questions_local, load_pubmed_passages, load_standard_dataset
from src.eval.summary import summarize_results
from src.LinearRAG import LinearRAG
from src.llm import LLM


def _ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path


def _default_run_name(args) -> str:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    if getattr(args, "use_mirage", False):
        ds = "_".join(getattr(args, "mirage_dataset", []) or ["unknown"])
        return f"{args.dataset_name}_mirage_{ds}_{ts}"
    return f"{args.dataset_name}_{ts}"


def _choose_index_namespace(args) -> str:
    """
    决定 LinearRAGConfig.dataset_name（对应 import/<dataset_name>/ 下的缓存/索引）。
    """
    if getattr(args, "reuse_index", None):
        return args.reuse_index

    if getattr(args, "use_mirage", False):
        base = f"{args.dataset_name}_mirage_{'_'.join(args.mirage_dataset)}"

        # 经验规则：bioasq/pubmedqa 往往复用同一套 PubMed 语料与图（比如 medqa 的 50k/10k 索引）
        if any(ds in set([d.lower() for d in args.mirage_dataset]) for ds in ["bioasq", "pubmedqa"]):
            medqa_ns = f"{args.dataset_name}_mirage_medqa"
            if os.path.exists(os.path.join("import", medqa_ns)):
                return medqa_ns

        return base

    return args.dataset_name


def run(args) -> Tuple[List[Dict[str, Any]], Dict[str, Any], str, str]:
    """
    执行一次完整流程：
    1) load embedding
    2) load passages + questions
    3) build/reuse index (import/<namespace>/)
    4) rag_model.qa
    5) summarize + save results

    Returns:
      (results, summary, results_path, summary_path)
    """
    out_root = _ensure_dir(getattr(args, "output_dir", "experiments"))
    results_dir = _ensure_dir(os.path.join(out_root, "results"))

    run_name = getattr(args, "run_name", None) or _default_run_name(args)

    # embedding model
    embedding_model = SentenceTransformer(args.embedding_model)
    embedding_model.eval()

    # questions + passages
    if getattr(args, "use_mirage", False):
        questions = load_mirage_questions_local(args.mirage_dataset, args.questions_limit)
        passages = load_pubmed_passages(chunks_limit=args.chunks_limit)
    else:
        questions, passages = load_standard_dataset(args.dataset_name)

    if not questions:
        raise RuntimeError("No questions loaded. Please check dataset path / loader.")
    if not passages:
        raise RuntimeError("No passages loaded. Please check corpus path / loader.")

    # llm
    llm_model = LLM(model_name=args.llm_model)

    # config / index namespace
    index_namespace = _choose_index_namespace(args)
    config = LinearRAGConfig(
        embedding_model=embedding_model,
        dataset_name=index_namespace,
        spacy_model=args.spacy_model,
        max_workers=args.max_workers,
        llm_model=llm_model,
        use_hf_ner=args.use_hf_ner or args.use_enhanced_ner,
        use_enhanced_ner=args.use_enhanced_ner,
        working_dir="import",
    )

    rag_model = LinearRAG(global_config=config)
    rag_model.index(passages)

    results = rag_model.qa(questions)
    summary = summarize_results(results)

    results_path = os.path.join(results_dir, f"{run_name}.json")
    summary_path = os.path.join(results_dir, f"{run_name}_summary.json")

    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    return results, summary, results_path, summary_path


