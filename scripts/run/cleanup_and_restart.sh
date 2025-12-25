#!/bin/bash

echo "================================================"
echo "Step 1: Stopping running processes"
echo "================================================"

# 查找并杀死运行中的 Python 进程
pkill -f "python run.py"
pkill -f "run.py"

# 等待进程完全停止
sleep 5

echo "✓ Processes stopped"

echo ""
echo "================================================"
echo "Step 2: Cleaning up processed data"
echo "================================================"

# 删除已处理的数据（保留原始数据）
rm -rf working_dir/pubmed/passage_embedding.parquet
rm -rf working_dir/pubmed/entity_embedding.parquet
rm -rf working_dir/pubmed/sentence_embedding.parquet
rm -rf working_dir/pubmed/ner_results.json
rm -rf working_dir/pubmed/LinearRAG.graphml

echo "✓ Cleaned up processed data"

echo ""
echo "================================================"
echo "Step 3: Verifying cleanup"
echo "================================================"

if [ -d "working_dir/pubmed" ]; then
    echo "Remaining files in working_dir/pubmed:"
    ls -lh working_dir/pubmed/
else
    echo "working_dir/pubmed does not exist"
fi

echo ""
echo "================================================"
echo "Cleanup completed!"
echo "================================================"
