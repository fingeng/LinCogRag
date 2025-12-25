#!/bin/bash

# 测试：仅使用稠密检索（跳过图搜索）

echo "============================================================"
echo "Test: Dense Retrieval Only (No Graph Search)"
echo "============================================================"

# 使用已经构建好的图，只重新运行测试
echo "Running test on 50 questions with dense retrieval only..."
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
echo "Test complete! Check results:"
echo "  results_pubmed_pubmedqa_pubmedqa.json"
echo "  results_pubmed_pubmedqa_pubmedqa_summary.json"
echo ""
echo "Expected with dense retrieval:"
echo "  - Higher retrieval scores (0.1-0.3 vs 0.002-0.008)"
echo "  - Correct documents retrieved"
echo "  - Better accuracy (hopefully 60-80%)"
echo "============================================================"
