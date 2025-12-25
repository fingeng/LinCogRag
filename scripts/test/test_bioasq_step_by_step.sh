#!/bin/bash

# 最简单的BioASQ测试 - 单步调试

source ~/miniconda3/bin/activate medgraphrag

cd /home/maoxy23/projects/LinearRAG

echo "测试1: 检查BioASQ数据加载..."
python -c "
from run import load_bioasq
questions = []
count = load_bioasq('MIRAGE/rawdata/bioasq', questions, 5)
print(f'✅ 成功加载 {count} 个问题')
print(f'第一个问题: {questions[0][\"question\"][:80]}...')
print(f'答案: {questions[0][\"answer\"]}')
"

echo ""
echo "测试2: 检查50k图是否存在..."
if [ -d "import/pubmed_mirage_medqa" ]; then
    echo "✅ 图目录存在"
    ls -lh import/pubmed_mirage_medqa/*.pkl 2>/dev/null | head -3
else
    echo "❌ 图目录不存在!"
    exit 1
fi

echo ""
echo "测试3: 运行完整测试(5个问题)..."
echo "开始时间: $(date)"

python run.py \
    --use_mirage \
    --mirage_dataset bioasq \
    --questions_limit 5 \
    --embedding_model model/all-mpnet-base-v2

echo ""
echo "结束时间: $(date)"
echo ""
echo "检查结果文件..."
if [ -f "results/pubmed_mirage_bioasq/evaluation_results.json" ]; then
    echo "✅ 结果文件已生成"
    python -c "
import json
with open('results/pubmed_mirage_bioasq/evaluation_results.json', 'r') as f:
    data = json.load(f)
print(f'准确率: {data.get(\"accuracy\", 0):.2%}')
print(f'平均时间: {data.get(\"avg_time_per_question\", 0):.2f}s')
print(f'总问题数: {data.get(\"total_questions\", 0)}')
"
else
    echo "❌ 结果文件未生成"
fi
