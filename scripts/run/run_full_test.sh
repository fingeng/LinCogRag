#!/bin/bash

echo "================================================"
echo "Full Test - All 5 Medical QA Datasets"
echo "================================================"
echo "This will:"
echo "1. Build graph on 1000 passages (~15 min)"
echo "2. Test on ALL questions from 5 datasets"
echo "Using Hybrid NER: BC5CDR + HuggingFace"
echo "Estimated time: ~40-60 minutes total"
echo "================================================"
echo ""

python run.py \
  --embedding_model model/all-mpnet-base-v2 \
  --dataset_name pubmed \
  --llm_model gpt-4o-mini \
  --max_workers 8 \
  --chunks_limit 1000 \
  --use_mirage \
  --mirage_dataset medqa mmlu medmcqa pubmedqa bioasq \
  > full_test.log 2>&1 &

PID=$!
echo "Started process with PID: $PID"
echo "Log file: full_test.log"
echo ""
echo "To monitor progress:"
echo "  tail -f full_test.log | grep -E '✅|Dataset|Accuracy'"
echo ""
echo "To check if still running:"
echo "  ps aux | grep $PID"
echo ""
echo "To stop:"
echo "  kill $PID"
