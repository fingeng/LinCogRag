#!/bin/bash

echo "================================================"
echo "MedQA Test with Hybrid NER"
echo "================================================"

# 删除旧缓存
echo "Cleaning old cache..."
rm -rf import/pubmed_mirage_medqa/
rm -rf working_dir/pubmed_mirage_medqa/

echo "Starting MedQA test..."
echo "Strategy: BC5CDR + HuggingFace (Hybrid)"
echo "Processing 1000 passages + ALL MedQA questions"
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
echo "Monitor: tail -f medqa_test.log | grep -E '✅|⚠️|Step|Accuracy'"
