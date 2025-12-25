#!/bin/bash
#
# MedMCQA 完整测试脚本
# 使用已有的 pubmed_mirage_medqa 图索引
#

echo "============================================"
echo "MedMCQA Full Test"
echo "Dataset: MIRAGE/rawdata/medmcqa/data/dev.json"
echo "Graph: pubmed_mirage_medqa (10k chunks)"
echo "Start time: $(date)"
echo "============================================"

# ✅ 限制并发数量避免SSH断连
nohup python run.py \
  --use_hf_ner \
  --embedding_model model/all-mpnet-base-v2 \
  --dataset_name pubmed \
  --llm_model gpt-4o-mini \
  --max_workers 4 \
  --use_mirage \
  --mirage_dataset medmcqa \
  --chunks_limit 10000 \
  > medmcqa_full_$(date +%Y%m%d_%H%M%S).log 2>&1 &

PID=$!
echo "============================================"
echo "Job started in background"
echo "Process ID: $PID"
echo "Monitor with: tail -f medmcqa_full_*.log"
echo "============================================"

# 保存 PID
echo $PID > medmcqa_test.pid

echo ""
echo "💡 监控命令:"
echo "  ./monitor_medmcqa.sh"
echo "  tail -f medmcqa_full_*.log"
echo ""
