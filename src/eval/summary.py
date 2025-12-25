from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Tuple


def _norm(s: Any) -> str:
    if s is None:
        return ""
    return str(s).strip()


def _norm_lower(s: Any) -> str:
    return _norm(s).lower()


def _valid_set_for_dataset(dataset: str) -> set[str]:
    ds = (dataset or "unknown").lower()
    if ds in {"pubmedqa"}:
        return {"yes", "no", "maybe"}
    if ds in {"bioasq"}:
        return {"yes", "no"}
    # default: 4-option MCQ
    return {"a", "b", "c", "d"}


def summarize_results(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    统一评测统计：
    - LLM Accuracy：pred == gold（按数据集规则规范化）
    - Invalid answers：pred 不在该数据集允许集合内（MCQ: A-D, PubMedQA: Yes/No/Maybe, BioASQ: Yes/No）
    - Contain Accuracy（弱定义）：对 MCQ，尽量判断“gold 选项文本”是否出现在检索 passages 中；其他数据集返回 0/不可解释但仍统计。
    """
    dataset_stats = defaultdict(
        lambda: {
            "total": 0,
            "correct": 0,
            "invalid": 0,
            "contain_correct": 0,
            "no_entity": 0,
        }
    )

    invalid_samples: List[Dict[str, Any]] = []

    for idx, r in enumerate(results):
        dataset = (r.get("dataset") or "unknown").lower()
        valid = _valid_set_for_dataset(dataset)

        pred_raw = _norm(r.get("pred_answer"))
        gold_raw = _norm(r.get("answer") or r.get("gold_answer"))

        pred_l = pred_raw.lower()
        gold_l = gold_raw.lower()

        dataset_stats[dataset]["total"] += 1

        # no entity
        if not r.get("has_entities", True):
            dataset_stats[dataset]["no_entity"] += 1

        # invalid
        if pred_l not in valid:
            dataset_stats[dataset]["invalid"] += 1
            invalid_samples.append(
                {
                    "index": idx,
                    "dataset": dataset,
                    "pred_answer": pred_raw[:120],
                    "question": _norm(r.get("question"))[:200],
                }
            )

        # accuracy
        if pred_l == gold_l and pred_l in valid:
            dataset_stats[dataset]["correct"] += 1

        # contain accuracy (best-effort)
        contain_found = False
        passages = r.get("sorted_passage") or []
        if passages and isinstance(passages, list):
            joined = " ".join([_norm(p) for p in passages]).lower()
            if dataset in {"mmlu", "medqa", "medmcqa"}:
                # 从题干提取 gold 选项文本（形如 "A. xxx"）
                qtext = _norm(r.get("question"))
                option_text = ""
                for line in qtext.splitlines():
                    line_s = line.strip()
                    if line_s.lower().startswith(f"{gold_l}."):
                        option_text = line_s.split(".", 1)[1].strip()
                        break
                if option_text and option_text.lower() in joined:
                    contain_found = True
            else:
                # 对 yes/no/maybe：弱定义（仅检查 gold 字符串是否出现）
                if gold_l and gold_l in joined:
                    contain_found = True

        if contain_found:
            dataset_stats[dataset]["contain_correct"] += 1

    # overall
    total_questions = sum(v["total"] for v in dataset_stats.values())
    total_correct = sum(v["correct"] for v in dataset_stats.values())
    total_invalid = sum(v["invalid"] for v in dataset_stats.values())
    total_contain_correct = sum(v["contain_correct"] for v in dataset_stats.values())
    total_no_entity = sum(v["no_entity"] for v in dataset_stats.values())

    overall_llm_acc = (total_correct / total_questions * 100) if total_questions else 0.0
    overall_contain_acc = (total_contain_correct / total_questions * 100) if total_questions else 0.0
    valid_answer_rate = ((total_questions - total_invalid) / total_questions * 100) if total_questions else 0.0

    return {
        "total_questions": total_questions,
        "overall_llm_accuracy": overall_llm_acc,
        "overall_contain_accuracy": overall_contain_acc,
        "total_correct": total_correct,
        "total_contain_correct": total_contain_correct,
        "total_invalid": total_invalid,
        "valid_answer_rate": valid_answer_rate,
        "questions_wo_entities": total_no_entity,
        "dataset_stats": {k: dict(v) for k, v in dataset_stats.items()},
        "invalid_samples": invalid_samples[:20],  # 仅保留前20条便于调试
    }


