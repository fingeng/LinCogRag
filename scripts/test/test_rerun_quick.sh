#!/bin/bash

# 快速诊断：重新测试50个问题（图已存在，应该很快）

echo "============================================================"
echo "Diagnostic Test: Re-run with existing graph"
echo "============================================================"

echo "Testing 50 questions with existing graph..."
python run.py \
    --dataset_name pubmed_pubmedqa \
    --embedding_model model/all-mpnet-base-v2 \
    --use_mirage \
    --mirage_dataset pubmedqa \
    --llm_model gpt-4o-mini \
    --use_hf_ner \
    --questions_limit 50

echo ""
echo "============================================================"
echo "Done! Now check if graph search works correctly."
echo "============================================================"
