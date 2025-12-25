#!/bin/bash

echo "================================================"
echo "Clean Start - MedQA Test with Hybrid NER"
echo "================================================"

# 🔧 删除所有缓存
echo "Cleaning all caches..."
rm -rf import/pubmed_mirage_medqa/
rm -rf working_dir/pubmed_mirage_medqa/

echo ""
echo "Starting fresh MedQA test..."
echo "Strategy: BC5CDR + HuggingFace (Hybrid)"
echo "Processing 1000 passages + ALL MedQA questions (1273)"
echo "================================================"

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python run.py \
  --use_hf_ner \
  --embedding_model model/all-mpnet-base-v2 \
  --dataset_name pubmed \
  --llm_model gpt-4o-mini \
  --max_workers 8 \
  --use_mirage \
  --mirage_dataset medqa \
  --chunks_limit 1000 \
  > medqa_test.log 2>&1 &

PID=$!
echo ""
echo "Started with PID: $PID"
echo "Log: medqa_test.log"
echo ""
echo "Monitor:"
echo "  tail -f medqa_test.log"
echo ""
echo "Check progress:"
echo "  tail -f medqa_test.log | grep -E '✅|⚠️|Step|Accuracy|Loaded'"
