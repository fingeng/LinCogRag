#!/bin/bash

# BioASQ Quick Test (5 questions)
# 快速验证BioASQ集成是否正确

# 激活conda环境
source ~/miniconda3/bin/activate medgraphrag

echo "======================================"
echo "BioASQ Quick Test (5 questions)"
echo "======================================"
echo ""

python run.py \
    --use_mirage \
    --mirage_dataset bioasq \
    --questions_limit 5 \
    --embedding_model model/all-mpnet-base-v2

echo ""
echo "======================================"
echo "Quick test completed!"
echo ""

if [ -f results/pubmed_mirage_bioasq/evaluation_results.json ]; then
    echo "📊 Results:"
    python -c "
import json
with open('results/pubmed_mirage_bioasq/evaluation_results.json', 'r') as f:
    data = json.load(f)
print(f\"  Accuracy: {data.get('accuracy', 0):.2%}\")
print(f\"  Avg time: {data.get('avg_time_per_question', 0):.2f}s\")
print()
print('Sample predictions:')
for i, result in enumerate(data.get('results', [])[:3]):
    print(f\"  Q{i+1}: Pred={result.get('predicted_answer')} | Gold={result.get('gold_answer')} | Correct={result.get('is_correct')}\")
"
fi
