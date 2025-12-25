#!/bin/bash

echo "================================================"
echo "Quick Test - MedQA Full Dataset"
echo "================================================"
echo "Testing with 1000 passages + ALL questions"
echo "Using Hybrid NER: BC5CDR + HuggingFace"
echo "Estimated time: ~20-30 minutes"
echo "================================================"
echo ""

python run.py \
  --embedding_model model/all-mpnet-base-v2 \
  --dataset_name pubmed \
  --llm_model gpt-4o-mini \
  --max_workers 8 \
  --chunks_limit 1000 \
  --use_mirage \
  --mirage_dataset medqa \
  > quick_test.log 2>&1 &

PID=$!
echo "Started quick test with PID: $PID"
echo "Log: quick_test.log"
echo ""
echo "Monitor progress:"
echo "  tail -f quick_test.log | grep -E '✅|⚠️|Step|Loaded|Results|Accuracy'"
echo ""
echo "Check if running:"
echo "  ps aux | grep $PID"
