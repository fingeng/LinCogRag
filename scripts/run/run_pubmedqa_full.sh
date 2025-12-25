#!/bin/bash
#
# PubMedQA 完整测试脚本 (500 Yes/No/Maybe问题)
# 复用已有的 pubmed_mirage_medqa 图索引 (50k chunks)
# 预计运行时间: ~15-20分钟 (500 questions × 1.9s/question)
#

# ✅ 激活环境和设置API
source ~/miniconda3/bin/activate medgraphrag
export OPENAI_BASE_URL="https://api.chatanywhere.tech"
export OPENAI_API_KEY="sk-RXbQMpzfo7ERxebnz9PTFQruIbAuBQ6odYPnrzaclBmG2vDc"

echo "============================================"
echo "PubMedQA Full Test"
echo "Dataset: MIRAGE/rawdata/pubmedqa (500 Yes/No/Maybe questions)"
echo "Graph: Reusing pubmed_mirage_medqa (50k chunks)"
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
  --mirage_dataset pubmedqa \
  --chunks_limit 50000 \
  > pubmedqa_full_$(date +%Y%m%d_%H%M%S).log 2>&1 &

PID=$!
echo "============================================"
echo "Job started in background"
echo "Process ID: $PID"
echo "Monitor with: tail -f pubmedqa_full_*.log"
echo "Or: watch -n 2 'tail -30 pubmedqa_full_*.log'"
echo "============================================"
