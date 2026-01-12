#!/usr/bin/env python3
"""
Quick validation script - test 50 questions per dataset to validate optimization effects.
"""
import sys
sys.path.insert(0, '/home/maoxy23/projects/LinearRAG')

import json
import os
from collections import defaultdict

# Settings
LLM_MODEL = "gpt-5-mini-ca"
QUESTIONS_LIMIT = 50  # 每个数据集50题快速验证
DATA_ROOT = "/home/maoxy23/projects/LinearRAG/MIRAGE/rawdata"
CHUNKS_DIR = "/home/maoxy23/projects/LinearRAG/dataset/pubmed/chunk"
CHUNKS_LIMIT = 20000

print("="*60)
print("🚀 Quick Validation Test (50 questions per dataset)")
print("="*60)

from src.config import LinearRAGConfig
from src.LinearRAG import LinearRAG
from sentence_transformers import SentenceTransformer

# Initialize embedding model
EMBEDDING_MODEL_PATH = "model/all-mpnet-base-v2"
embedding_model = SentenceTransformer(EMBEDDING_MODEL_PATH)

# Initialize config
config = LinearRAGConfig(
    embedding_model=embedding_model,
    dataset_name="medqa",  # default
    llm_model=LLM_MODEL,
    use_hypergraph=True
)
linearRAG = LinearRAG(config)

# Load chunks
print("\n[1/3] Loading chunks...")
import glob
chunks = []
jsonl_files = sorted(glob.glob(os.path.join(CHUNKS_DIR, "pubmed23n*.jsonl")))
print(f"Found {len(jsonl_files)} chunk files")
for file_path in jsonl_files:
    if len(chunks) >= CHUNKS_LIMIT:
        break
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            if len(chunks) >= CHUNKS_LIMIT:
                break
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
                text = item.get("contents") or item.get("text") or item.get("content") or ""
                text = text.strip()
                if text:
                    chunks.append({"contents": text})
            except json.JSONDecodeError:
                continue
print(f"Loaded {len(chunks)} chunks")
# 转换为text列表
passages = [c.get("contents", "") for c in chunks if c.get("contents")]
print(f"Indexing {len(passages)} passages...")
linearRAG.index(passages)

# Load datasets - take balanced samples
def load_balanced_dataset(dataset_name, limit):
    """Load dataset with balanced answer distribution."""
    questions = []
    
    if dataset_name == "pubmedqa":
        path = os.path.join(DATA_ROOT, "pubmedqa/data/test_set.json")
        with open(path, 'r') as f:
            data = json.load(f)
        
        # Group by answer
        by_answer = {'yes': [], 'no': [], 'maybe': []}
        for k, v in data.items():
            ans = v.get('final_decision', '').lower()
            if ans in by_answer:
                by_answer[ans].append({
                    "question": v['QUESTION'],
                    "answer": ans.capitalize(),
                    "dataset": dataset_name
                })
        
        # Take balanced samples: proportional to full distribution (55:34:11)
        n_yes = int(limit * 0.55)
        n_no = int(limit * 0.34)
        n_maybe = limit - n_yes - n_no
        
        questions.extend(by_answer['yes'][:n_yes])
        questions.extend(by_answer['no'][:n_no])
        questions.extend(by_answer['maybe'][:n_maybe])
        print(f"  PubMedQA: Yes={len(by_answer['yes'][:n_yes])}, No={len(by_answer['no'][:n_no])}, Maybe={len(by_answer['maybe'][:n_maybe])}")
        
    elif dataset_name == "bioasq":
        import glob
        task_dirs = sorted(glob.glob(os.path.join(DATA_ROOT, "bioasq/Task*BGoldenEnriched")))
        by_answer = {'yes': [], 'no': []}
        
        for task_dir in task_dirs:
            json_files = sorted(glob.glob(os.path.join(task_dir, "*_golden.json")))
            for jf in json_files:
                try:
                    with open(jf, 'r') as f:
                        d = json.load(f)
                    for q in d.get('questions', []):
                        if q.get('type', '').lower() == 'yesno':
                            ans = q.get('exact_answer', '').lower()
                            if ans in ['yes', 'no']:
                                by_answer[ans].append({
                                    "question": q['body'],
                                    "answer": ans.capitalize(),
                                    "dataset": dataset_name
                                })
                except:
                    continue
        
        # Balanced: 64% Yes, 36% No
        n_yes = int(limit * 0.64)
        n_no = limit - n_yes
        questions.extend(by_answer['yes'][:n_yes])
        questions.extend(by_answer['no'][:n_no])
        print(f"  BioASQ: Yes={len(by_answer['yes'][:n_yes])}, No={len(by_answer['no'][:n_no])}")
    
    return questions

def load_mcq_dataset(dataset_name, limit):
    """Load MCQ dataset."""
    questions = []
    
    if dataset_name == "medqa":
        path = os.path.join(DATA_ROOT, "medqa/data_clean/questions/US/test.jsonl")
        with open(path, 'r') as f:
            for i, line in enumerate(f):
                if i >= limit:
                    break
                d = json.loads(line)
                options = d.get('options', {})
                q_text = d['question'] + "\n"
                for k, v in sorted(options.items()):
                    q_text += f"({k}) {v}\n"
                questions.append({
                    "question": q_text,
                    "answer": d['answer_idx'],
                    "dataset": dataset_name
                })
                
    elif dataset_name == "medmcqa":
        path = os.path.join(DATA_ROOT, "medmcqa/dev.json")
        with open(path, 'r') as f:
            data = json.load(f)
        for i, item in enumerate(data[:limit]):
            q_text = item['question'] + "\n"
            for k, v in [('A', 'opa'), ('B', 'opb'), ('C', 'opc'), ('D', 'opd')]:
                q_text += f"({k}) {item.get(v, '')}\n"
            answer_map = {0: 'A', 1: 'B', 2: 'C', 3: 'D'}
            questions.append({
                "question": q_text,
                "answer": answer_map.get(item.get('cop', 0), 'A'),
                "dataset": dataset_name
            })
            
    elif dataset_name == "mmlu":
        path = os.path.join(DATA_ROOT, "mmlu/data")
        medical_subsets = [
            "anatomy", "clinical_knowledge", "college_biology", "college_medicine",
            "medical_genetics", "professional_medicine"
        ]
        for subset in medical_subsets:
            test_file = os.path.join(path, subset, f"test_{subset}.csv")
            if os.path.exists(test_file):
                with open(test_file, 'r') as f:
                    for line in f:
                        parts = line.strip().split(',')
                        if len(parts) >= 6:
                            question = parts[0]
                            options = parts[1:5]
                            answer = parts[5]
                            q_text = question + "\n"
                            for j, opt in enumerate(options):
                                q_text += f"({chr(65+j)}) {opt}\n"
                            questions.append({
                                "question": q_text,
                                "answer": answer,
                                "dataset": dataset_name
                            })
                        if len(questions) >= limit:
                            break
            if len(questions) >= limit:
                break
        questions = questions[:limit]
    
    return questions

# Load all datasets
print("\n[2/3] Loading balanced test data...")
all_questions = []
all_questions.extend(load_balanced_dataset("pubmedqa", QUESTIONS_LIMIT))
all_questions.extend(load_balanced_dataset("bioasq", QUESTIONS_LIMIT))
all_questions.extend(load_mcq_dataset("medqa", QUESTIONS_LIMIT))
all_questions.extend(load_mcq_dataset("medmcqa", QUESTIONS_LIMIT))
all_questions.extend(load_mcq_dataset("mmlu", QUESTIONS_LIMIT))

print(f"\nTotal: {len(all_questions)} questions")

# Run QA
print("\n[3/3] Running QA...")
results = linearRAG.qa(all_questions)

# Analyze results
print("\n" + "="*60)
print("📊 Results Analysis")
print("="*60)

datasets = ['medqa', 'medmcqa', 'mmlu', 'pubmedqa', 'bioasq']
targets = {'medqa': 82.80, 'medmcqa': 66.65, 'mmlu': 87.24, 'pubmedqa': 70.60, 'bioasq': 92.56}

print("\n| Dataset | Correct | Total | Accuracy | Target | Gap |")
print("|---------|---------|-------|----------|--------|-----|")

total_correct = 0
total_count = 0

for ds in datasets:
    ds_results = [r for r in results if r.get('dataset') == ds]
    if not ds_results:
        continue
    
    correct = sum(1 for r in ds_results if str(r.get('pred_answer', '')).upper() == str(r.get('answer', '')).upper())
    total = len(ds_results)
    acc = correct / total * 100 if total > 0 else 0
    target = targets[ds]
    gap = acc - target
    
    total_correct += correct
    total_count += total
    
    sign = "+" if gap >= 0 else ""
    status = "✅" if gap >= 0 else "❌"
    print(f"| {ds:8} | {correct:7} | {total:5} | {acc:6.1f}%  | {target:5.1f}% | {sign}{gap:+.1f}% {status} |")

overall_acc = total_correct / total_count * 100 if total_count > 0 else 0
print(f"| {'TOTAL':8} | {total_correct:7} | {total_count:5} | {overall_acc:6.1f}%  | 79.97% | {overall_acc-79.97:+.1f}% |")

# Detailed analysis for Yes/No datasets
print("\n" + "="*60)
print("Yes/No Dataset Detailed Analysis")
print("="*60)

for ds in ['pubmedqa', 'bioasq']:
    ds_results = [r for r in results if r.get('dataset') == ds]
    if not ds_results:
        continue
    
    pred_dist = defaultdict(int)
    gold_dist = defaultdict(int)
    confusion = defaultdict(int)
    
    for r in ds_results:
        pred = str(r.get('pred_answer', 'INVALID')).lower()
        gold = str(r.get('answer', '')).lower()
        pred_dist[pred] += 1
        gold_dist[gold] += 1
        confusion[f"{gold}->{pred}"] += 1
    
    print(f"\n{ds.upper()}:")
    print(f"  Prediction distribution: {dict(pred_dist)}")
    print(f"  Gold distribution: {dict(gold_dist)}")
    print(f"  Confusion: {dict(confusion)}")

print("\n✅ Quick validation complete!")
