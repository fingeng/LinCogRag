from __future__ import annotations

import csv
import glob
import json
import os
from typing import Dict, List, Optional, Sequence


def load_pubmed_passages(chunks_dir: str = "dataset/pubmed/chunk", chunks_limit: Optional[int] = None) -> List[str]:
    """
    从 PubMed JSONL chunk 文件加载 passages。
    注意：为了复用已有索引/NER 缓存，这里不对 passage 文本做额外前缀加工（hash_id 依赖原始 text）。
    """
    passages: List[str] = []
    if not os.path.exists(chunks_dir):
        raise FileNotFoundError(f"PubMed chunks dir not found: {chunks_dir}")

    jsonl_files = sorted(glob.glob(os.path.join(chunks_dir, "*.jsonl")))
    if not jsonl_files:
        raise FileNotFoundError(f"No JSONL files found in {chunks_dir}")

    for file_path in jsonl_files:
        if chunks_limit is not None and len(passages) >= chunks_limit:
            break
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                for line in f:
                    if chunks_limit is not None and len(passages) >= chunks_limit:
                        break
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        item = json.loads(line)
                    except json.JSONDecodeError:
                        continue

                    # 兼容不同字段名
                    text = item.get("contents") or item.get("text") or item.get("content") or item.get("passage") or ""
                    text = text.strip()
                    if text:
                        passages.append(text)
        except Exception:
            # 单个文件损坏不应中断整次运行
            continue

    return passages


def load_standard_dataset(dataset_name: str) -> tuple[List[Dict], List[str]]:
    """
    兼容非 MIRAGE 模式：读取 dataset/<name>/questions.json & chunks.json
    """
    questions_path = os.path.join("dataset", dataset_name, "questions.json")
    chunks_path = os.path.join("dataset", dataset_name, "chunks.json")
    with open(questions_path, "r", encoding="utf-8") as f:
        questions = json.load(f)
    with open(chunks_path, "r", encoding="utf-8") as f:
        chunks = json.load(f)
    passages = [f"{idx}:{chunk}" for idx, chunk in enumerate(chunks)]
    return questions, passages


def load_mirage_questions_local(
    mirage_dataset: Sequence[str],
    questions_limit: Optional[int] = None,
    mirage_root: str = "MIRAGE/rawdata",
) -> List[Dict]:
    """
    统一本地 MIRAGE rawdata 五数据集接口：
    返回 list[dict]，字段统一为：
    - question: str（对四选一，会把 options 拼进 question 文本里，便于 LLM 强制输出字母）
    - answer: str（MCQ: A/B/C/D；pubmedqa: yes/no/maybe；bioasq: yes/no）
    - dataset: str（medqa/medmcqa/mmlu/pubmedqa/bioasq）
    """
    datasets = list(mirage_dataset)
    all_questions: List[Dict] = []
    for ds in datasets:
        ds_l = ds.lower()
        if ds_l == "medqa":
            path = os.path.join(mirage_root, "medqa", "data_clean", "questions", "US", "4_options", "phrases_no_exclude_test.jsonl")
            all_questions.extend(_load_medqa_jsonl(path, questions_limit))
        elif ds_l == "medmcqa":
            path = os.path.join(mirage_root, "medmcqa", "data", "dev.json")
            all_questions.extend(_load_medmcqa_json_or_jsonl(path, questions_limit))
        elif ds_l == "pubmedqa":
            path = os.path.join(mirage_root, "pubmedqa", "data", "test_set.json")
            all_questions.extend(_load_pubmedqa_json(path, questions_limit))
        elif ds_l == "bioasq":
            path = os.path.join(mirage_root, "bioasq")
            all_questions.extend(_load_bioasq_yesno(path, questions_limit))
        elif ds_l == "mmlu":
            path = os.path.join(mirage_root, "mmlu", "data", "test")
            all_questions.extend(_load_mmlu_csv_dir(path, questions_limit))
        else:
            raise ValueError(f"Unknown mirage dataset: {ds}")
    return all_questions


def _load_medqa_jsonl(dataset_file: str, questions_limit: Optional[int]) -> List[Dict]:
    if not os.path.exists(dataset_file):
        raise FileNotFoundError(f"MedQA file not found: {dataset_file}")
    out: List[Dict] = []
    with open(dataset_file, "r", encoding="utf-8") as f:
        for line in f:
            if questions_limit is not None and len(out) >= questions_limit:
                break
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue

            question_text = (item.get("question") or "").strip()
            answer_idx = (item.get("answer_idx") or "").strip()
            options = item.get("options") or {}
            if not question_text or not answer_idx:
                continue

            full_question = question_text + "\n\nOptions:\n"
            for key in sorted(options.keys()):
                full_question += f"{key}. {options[key]}\n"

            out.append({"question": full_question, "answer": answer_idx, "dataset": "medqa"})
    return out


def _load_medmcqa_json_or_jsonl(dataset_file: str, questions_limit: Optional[int]) -> List[Dict]:
    if not os.path.exists(dataset_file):
        raise FileNotFoundError(f"MedMCQA file not found: {dataset_file}")
    out: List[Dict] = []

    # dev.json 在该仓库实际是 JSONL（逐行 JSON object）
    cop_mapping = {1: "A", 2: "B", 3: "C", 4: "D"}
    with open(dataset_file, "r", encoding="utf-8") as f:
        for line in f:
            if questions_limit is not None and len(out) >= questions_limit:
                break
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue

            question = (item.get("question") or "").strip()
            opa = (item.get("opa") or "").strip()
            opb = (item.get("opb") or "").strip()
            opc = (item.get("opc") or "").strip()
            opd = (item.get("opd") or "").strip()
            cop = item.get("cop")
            if not question:
                continue

            full_question = f"{question}\n\nA. {opa}\nB. {opb}\nC. {opc}\nD. {opd}"
            correct_answer = cop_mapping.get(cop, "A")
            out.append({"question": full_question, "answer": correct_answer, "dataset": "medmcqa"})
    return out


def _load_pubmedqa_json(dataset_file: str, questions_limit: Optional[int]) -> List[Dict]:
    if not os.path.exists(dataset_file):
        raise FileNotFoundError(f"PubMedQA file not found: {dataset_file}")
    with open(dataset_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    out: List[Dict] = []
    for pmid, item in data.items():
        if questions_limit is not None and len(out) >= questions_limit:
            break
        question = (item.get("QUESTION") or "").strip()
        answer = (item.get("final_decision") or "").strip().lower()
        if not question or answer not in {"yes", "no", "maybe"}:
            continue
        out.append({"question": question, "answer": answer, "dataset": "pubmedqa", "pmid": pmid})
    return out


def _load_bioasq_yesno(dataset_dir: str, questions_limit: Optional[int]) -> List[Dict]:
    if not os.path.exists(dataset_dir):
        raise FileNotFoundError(f"BioASQ dir not found: {dataset_dir}")
    out: List[Dict] = []
    task_dirs = sorted(glob.glob(os.path.join(dataset_dir, "Task*BGoldenEnriched")))
    for task_dir in task_dirs:
        json_files = sorted(glob.glob(os.path.join(task_dir, "*_golden.json")))
        for json_file in json_files:
            if questions_limit is not None and len(out) >= questions_limit:
                return out
            try:
                with open(json_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
            except Exception:
                continue

            for q in data.get("questions", []):
                if questions_limit is not None and len(out) >= questions_limit:
                    return out
                if (q.get("type") or "").lower() != "yesno":
                    continue
                body = (q.get("body") or "").strip()
                exact = q.get("exact_answer")
                ans = (exact or "").strip().lower() if isinstance(exact, str) else ""
                if not body or ans not in {"yes", "no"}:
                    continue
                out.append({"question": body, "answer": ans, "dataset": "bioasq", "id": q.get("id")})
    return out


def _load_mmlu_csv_dir(test_dir: str, questions_limit: Optional[int]) -> List[Dict]:
    if not os.path.exists(test_dir):
        raise FileNotFoundError(f"MMLU test dir not found: {test_dir}")
    out: List[Dict] = []
    csv_files = sorted(glob.glob(os.path.join(test_dir, "*_test.csv")))
    for csv_file in csv_files:
        if questions_limit is not None and len(out) >= questions_limit:
            break
        with open(csv_file, "r", encoding="utf-8") as f:
            reader = csv.reader(f)
            for row in reader:
                if questions_limit is not None and len(out) >= questions_limit:
                    break
                # 格式：question, A, B, C, D, answer_letter
                if len(row) < 6:
                    continue
                q, a, b, c, d, gold = row[0], row[1], row[2], row[3], row[4], row[5]
                q = (q or "").strip()
                gold = (gold or "").strip().upper()
                if not q or gold not in {"A", "B", "C", "D"}:
                    continue
                full_q = f"{q}\n\nA. {a}\nB. {b}\nC. {c}\nD. {d}"
                out.append({"question": full_q, "answer": gold, "dataset": "mmlu", "subject_file": os.path.basename(csv_file)})
    return out


