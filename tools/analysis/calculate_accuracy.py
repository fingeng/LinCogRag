import json
import sys
from collections import defaultdict

def calculate_accuracy(results_file):
    """Calculate accuracy from results file"""
    
    # Load results
    print(f"\n{'='*70}")
    print(f"Loading results from: {results_file}")
    print(f"{'='*70}\n")
    
    with open(results_file, 'r', encoding='utf-8') as f:
        results = json.load(f)
    
    # Statistics by dataset
    dataset_stats = defaultdict(lambda: {
        "total": 0,
        "correct": 0,
        "invalid": 0,
        "examples": []
    })
    
    # Analyze each result
    for idx, result in enumerate(results):
        dataset = result.get("dataset", "unknown")
        gold = result.get("gold_answer", "").strip().upper()
        pred = result.get("pred_answer", "").strip().upper()
        question = result.get("question", "")
        
        dataset_stats[dataset]["total"] += 1
        
        # Check if answer is valid (A/B/C/D for multiple choice)
        if pred not in ['A', 'B', 'C', 'D']:
            dataset_stats[dataset]["invalid"] += 1
            if len(dataset_stats[dataset]["examples"]) < 3:  # Keep first 3 invalid examples
                dataset_stats[dataset]["examples"].append({
                    "index": idx,
                    "question": question[:100] + "...",
                    "gold": gold,
                    "pred": pred[:100]
                })
        
        # Check correctness
        if pred == gold:
            dataset_stats[dataset]["correct"] += 1
    
    # Print results
    print(f"{'='*70}")
    print(f"📊 Accuracy Results")
    print(f"{'='*70}\n")
    
    total_correct = 0
    total_questions = 0
    total_invalid = 0
    
    for dataset in sorted(dataset_stats.keys()):
        stats = dataset_stats[dataset]
        correct = stats["correct"]
        total = stats["total"]
        invalid = stats["invalid"]
        accuracy = (correct / total * 100) if total > 0 else 0
        
        total_correct += correct
        total_questions += total
        total_invalid += invalid
        
        print(f"Dataset: {dataset}")
        print(f"  Total questions: {total}")
        print(f"  Correct answers: {correct}")
        print(f"  Invalid answers: {invalid}")
        print(f"  Accuracy: {accuracy:.2f}% ({correct}/{total})")
        
        # Show invalid examples
        if invalid > 0 and stats["examples"]:
            print(f"\n  ⚠️  Sample invalid answers:")
            for ex in stats["examples"]:
                print(f"    Q{ex['index']}: {ex['question']}")
                print(f"      Gold: {ex['gold']}, Predicted: {ex['pred']}")
        print()
    
    # Overall statistics
    overall_accuracy = (total_correct / total_questions * 100) if total_questions > 0 else 0
    
    print(f"{'='*70}")
    print(f"📈 Overall Statistics")
    print(f"{'='*70}")
    print(f"Total questions: {total_questions}")
    print(f"Correct answers: {total_correct}")
    print(f"Invalid answers: {total_invalid}")
    print(f"Valid answer rate: {((total_questions - total_invalid) / total_questions * 100):.2f}%")
    print(f"\n🎯 Overall Accuracy: {overall_accuracy:.2f}% ({total_correct}/{total_questions})")
    print(f"{'='*70}\n")
    
    # Additional analysis: Answer distribution
    print(f"{'='*70}")
    print(f"📊 Answer Distribution")
    print(f"{'='*70}")
    
    answer_dist = defaultdict(int)
    for result in results:
        pred = result.get("pred_answer", "").strip().upper()
        answer_dist[pred] += 1
    
    print(f"Predicted answer distribution:")
    for ans in sorted(answer_dist.keys()):
        count = answer_dist[ans]
        percentage = (count / total_questions * 100) if total_questions > 0 else 0
        print(f"  {ans}: {count} ({percentage:.1f}%)")
    print(f"{'='*70}\n")
    
    return {
        "overall_accuracy": overall_accuracy,
        "total_correct": total_correct,
        "total_questions": total_questions,
        "total_invalid": total_invalid,
        "dataset_stats": dict(dataset_stats)
    }

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python calculate_accuracy.py <results_file.json>")
        print("Example: python calculate_accuracy.py results_pubmed_medqa.json")
        sys.exit(1)
    
    results_file = sys.argv[1]
    
    try:
        stats = calculate_accuracy(results_file)
        
        # Save summary
        summary_file = results_file.replace('.json', '_summary.json')
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(stats, f, indent=2, ensure_ascii=False, default=str)
        
        print(f"💾 Summary saved to: {summary_file}")
        
    except FileNotFoundError:
        print(f"❌ Error: File not found: {results_file}")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
