#!/bin/bash

# 快速测试：只用500个PubMedQA CONTEXTS构建图

echo "============================================================"
echo "Quick Test: PubMedQA with CONTEXTS corpus"
echo "============================================================"

# Step 1: 准备corpus目录
echo "Step 1: Preparing corpus..."
mkdir -p dataset/pubmed_pubmedqa/chunk
cp pubmedqa_contexts_chunks.jsonl dataset/pubmed_pubmedqa/chunk/pubmed.jsonl

# Step 2: 构建图（500 passages）
echo ""
echo "Step 2: Building graph (this will take 5-10 minutes)..."
python run.py \
    --dataset_name pubmed_pubmedqa \
    --embedding_model model/all-mpnet-base-v2 \
    --use_mirage \
    --mirage_dataset pubmedqa \
    --llm_model gpt-4o-mini \
    --use_hf_ner \
    --chunks_limit 500

# Step 3: 运行测试（前50个问题）
echo ""
echo "Step 3: Running test on first 50 questions..."
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
echo "Test complete! Check the results."
echo "Expected improvements:"
echo "  - Retrieval scores: 0.001 -> 0.1-0.3 (100x-300x better)"
echo "  - Accuracy: ~0% -> 60-80%"
echo "============================================================"
