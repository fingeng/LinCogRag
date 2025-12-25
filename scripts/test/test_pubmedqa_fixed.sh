#!/bin/bash

# 修复版测试：确保使用500个CONTEXTS corpus，top-k=3

echo "============================================================"
echo "Fixed Test: PubMedQA with CONTEXTS (Top-K=3)"
echo "============================================================"

# 清理旧的图
echo "Cleaning old graph..."
rm -rf import/pubmed_pubmedqa_mirage_pubmedqa/

# Step 1: 准备corpus（确保位置正确）
echo ""
echo "Step 1: Preparing corpus..."
mkdir -p dataset/pubmed_pubmedqa/chunk
cp pubmedqa_contexts_chunks.jsonl dataset/pubmed_pubmedqa/chunk/pubmed.jsonl
echo "✅ Corpus ready: $(wc -l < dataset/pubmed_pubmedqa/chunk/pubmed.jsonl) chunks"

# Step 2: 一次性构建图并测试50个问题（top-k=3）
echo ""
echo "Step 2: Building graph and testing (top-k=3, set in config)..."
python run.py \
    --dataset_name pubmed_pubmedqa \
    --embedding_model model/all-mpnet-base-v2 \
    --use_mirage \
    --mirage_dataset pubmedqa \
    --llm_model gpt-4o-mini \
    --use_hf_ner \
    --chunks_limit 500 \
    --questions_limit 50

echo ""
echo "============================================================"
echo "Test complete! Results saved to:"
echo "  - results_pubmed_pubmedqa_pubmedqa.json"
echo "  - results_pubmed_pubmedqa_pubmedqa_summary.json"
echo ""
echo "Expected with top-k=3:"
echo "  - Less noise from irrelevant documents"
echo "  - Higher quality retrieval"
echo "  - Better accuracy if corpus is correct"
echo "============================================================"
