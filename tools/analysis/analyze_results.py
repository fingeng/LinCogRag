import json

# Load results
with open('results_pubmed_medqa.json', 'r') as f:
    results = json.load(f)

print("="*70)
print("📊 Results Analysis")
print("="*70)

# 1. Check answer format issues
invalid_count = 0
correct_count = 0
mismatch_examples = []

for i, result in enumerate(results):
    gold = result.get("answer", "").strip().upper()  # ✅ 使用 "answer" 而不是 "gold_answer"
    pred = result.get("pred_answer", "").strip().upper()
    raw = result.get("raw_answer", "").strip().upper()
    
    # Check if predicted answer is valid
    if pred not in ['A', 'B', 'C', 'D']:
        invalid_count += 1
    
    # Check if prediction matches gold
    if pred == gold:
        correct_count += 1
    
    # Check if raw != pred (parsing issue)
    if raw != pred and len(mismatch_examples) < 5:
        mismatch_examples.append({
            "index": i,
            "question": result["question"][:100] + "...",
            "gold": gold,
            "pred": pred,
            "raw": raw[:100]
        })

print(f"\n📈 Answer Statistics:")
print(f"  Total questions: {len(results)}")
print(f"  Correct predictions: {correct_count} ({correct_count/len(results)*100:.1f}%)")
print(f"  Invalid predictions: {invalid_count}")
print(f"  Accuracy: {correct_count/len(results)*100:.1f}%")

if mismatch_examples:
    print(f"\n⚠️  Raw answer != Parsed answer (first 5 cases):")
    for ex in mismatch_examples:
        print(f"\n  Q{ex['index']}: {ex['question']}")
        print(f"    Gold: {ex['gold']}, Pred: {ex['pred']}, Raw: {ex['raw']}")

# 2. Analyze entities (需要重新运行 retrieve 来统计)
print(f"\n{'='*70}")
print(f"💡 To see which questions have no entities:")
print(f"   Run with --questions_limit 50 and check log output")
print(f"{'='*70}\n")
