#!/bin/bash

source ~/miniconda3/bin/activate medgraphrag

# API配置：请先设置环境变量
if [ -z "$OPENAI_API_KEY" ]; then
    echo "❌ Error: OPENAI_API_KEY not set. Run: export OPENAI_API_KEY=\"your-key\""
    exit 1
fi
export OPENAI_BASE_URL="${OPENAI_BASE_URL:-https://api.openai.com/v1}"

cd /home/maoxy23/projects/LinearRAG

echo "=== 测试PubMedQA (3题) ==="
echo "开始时间: $(date '+%H:%M:%S')"
echo ""

python -u run.py \
  --use_hf_ner \
  --embedding_model model/all-mpnet-base-v2 \
  --dataset_name pubmed \
  --llm_model gpt-4o-mini \
  --max_workers 4 \
  --use_mirage \
  --mirage_dataset pubmedqa \
  --questions_limit 3 \
  --chunks_limit 50000 2>&1 | tee /tmp/pubmedqa_test.log

echo ""
echo "结束时间: $(date '+%H:%M:%S')"
