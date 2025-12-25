#!/bin/bash

echo "================================================"
echo "Monitoring LinearRAG Progress"
echo "================================================"
echo ""

# 检查进程是否还在运行
if pgrep -f "python run.py" > /dev/null; then
    echo "✓ Process is running"
    echo ""
    
    # 显示最新的进度
    echo "Recent progress:"
    echo "----------------"
    tail -n 50 full_test.log | grep -E "\[LinearRAG|Batch|Step|✅|⚠️"
    
    echo ""
    echo "Working directory status:"
    echo "------------------------"
    if [ -d "working_dir/pubmed" ]; then
        ls -lh working_dir/pubmed/ | grep -E "parquet|json|graphml"
    fi
else
    echo "✗ Process is not running"
    echo ""
    echo "Check log file for errors:"
    tail -n 100 full_test.log
fi
