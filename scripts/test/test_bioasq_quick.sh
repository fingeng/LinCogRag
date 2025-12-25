#!/bin/bash
#
# BioASQ 快速测试 (3题)
#

# ✅ 激活环境和设置API
source ~/miniconda3/bin/activate medgraphrag
export OPENAI_BASE_URL="https://api.chatanywhere.tech"
export OPENAI_API_KEY="sk-RXbQMpzfo7ERxebnz9PTFQruIbAuBQ6odYPnrzaclBmG2vDc"

echo "============================================"
echo "BioASQ Quick Test (3 questions)"
echo "Dataset: MIRAGE/rawdata/bioasq"
echo "Graph: pubmed_mirage_medqa (50k chunks)"
echo "Start time: $(date)"
echo "============================================"

python run.py \
  --use_hf_ner \
  --embedding_model model/all-mpnet-base-v2 \
  --dataset_name pubmed \
  --llm_model gpt-4o-mini \
  --max_workers 4 \
  --use_mirage \
  --mirage_dataset bioasq \
  --questions_limit 3 \
  --chunks_limit 50000

echo ""
echo "============================================"
echo "Test completed at: $(date)"
echo "============================================"

# 显示结果
if [ -f "results/pubmed_mirage_bioasq/evaluation_results.json" ]; then
    echo ""
    echo "📊 Results:"
    python3 << 'EOF'
import json
with open('results/pubmed_mirage_bioasq/evaluation_results.json', 'r') as f:
    data = json.load(f)
    
print(f"  Total questions: {data.get('total_questions', 0)}")
print(f"  Accuracy: {data.get('accuracy', 0)*100:.1f}%")
print(f"  Avg time: {data.get('avg_time_per_question', 0):.2f}s")

print("\n  Sample results:")
for i, r in enumerate(data.get('results', [])[:3]):
    correct = "✅" if r.get('is_correct') else "❌"
    print(f"  {i+1}. {correct} Pred: {r.get('predicted_answer')} | Gold: {r.get('gold_answer')}")
EOF
fi
