#!/bin/bash

echo "============================================"
echo "MedQA Full Test (All Questions)"
echo "Start time: $(date)"
echo "============================================"

# 使用已有的 10k chunks 索引
# ✅ 限制并发数量避免SSH断连
nohup python run.py \
  --use_hf_ner \
  --embedding_model model/all-mpnet-base-v2 \
  --dataset_name pubmed \
  --llm_model gpt-4o-mini \
  --max_workers 4 \
  --use_mirage \
  --mirage_dataset medqa \
  --chunks_limit 10000 \
  > medqa_full_$(date +%Y%m%d_%H%M%S).log 2>&1 &

PID=$!
echo "============================================"
echo "Job started in background"
echo "Process ID: $PID"
echo "Monitor with: tail -f medqa_full_*.log"
echo "============================================"

# 保存 PID
echo $PID > medqa_test.pid
