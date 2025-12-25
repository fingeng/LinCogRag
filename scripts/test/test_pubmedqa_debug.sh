#!/bin/bash

source ~/miniconda3/bin/activate medgraphrag
export OPENAI_BASE_URL="https://api.chatanywhere.tech"
export OPENAI_API_KEY="sk-RXbQMpzfo7ERxebnz9PTFQruIbAuBQ6odYPnrzaclBmG2vDc"

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
