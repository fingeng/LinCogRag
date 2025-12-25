#!/bin/bash

echo "============================================"
echo "MedQA Overnight Full Test"
echo "Start time: $(date)"
echo "============================================"

# 运行完整测试（全部 MedQA 问题 + 更多 chunks）
python run.py \
  --use_hf_ner \
  --embedding_model model/all-mpnet-base-v2 \
  --dataset_name pubmed \
  --llm_model gpt-4o-mini \
  --max_workers 8 \
  --use_mirage \
  --mirage_dataset medqa \
  --chunks_limit 50000 \
  --questions_limit -1 \
  > medqa_overnight_$(date +%Y%m%d_%H%M%S).log 2>&1

echo "============================================"
echo "Test completed!"
echo "End time: $(date)"
echo "Check results in results_pubmed_medqa.json"
echo "============================================"
