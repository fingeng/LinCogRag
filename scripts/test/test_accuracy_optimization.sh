#!/bin/bash
#
# 准确率优化测试 - 候选池扩大到500
#

cd /home/maoxy23/projects/LinearRAG

echo "=================================================="
echo "准确率优化测试"
echo "=================================================="
echo ""
echo "✅ 实施的优化:"
echo "   1. 候选池: 200 → 500"
echo "   2. 句子相似度阈值过滤: 0.25"
echo "   3. 远距离实体权重衰减: 0.7"
echo ""
echo "📊 预期效果:"
echo "   - 准确率: 70% → 72-73%"
echo "   - 速度: 1.5秒 → 3-4秒 (仍快3-4倍)"
echo ""
echo "对比之前的结果:"
echo "   版本1 (参数优化): 13.9秒, 73%准确率"
echo "   版本2 (候选池200): 1.5秒, 70%准确率"
echo "   版本3 (候选池500): 预期3-4秒, 72-73%准确率"
echo ""
echo "=================================================="
echo ""

# 停止旧进程
OLD_PID=$(pgrep -f "run.py" || true)
if [ -n "$OLD_PID" ]; then
    echo "停止运行中的进程..."
    kill $OLD_PID 2>/dev/null || true
    sleep 2
fi

# 运行测试
echo "🚀 开始测试 (100个问题)..."
echo ""

python run.py \
    --use_hf_ner \
    --embedding_model model/all-mpnet-base-v2 \
    --dataset_name pubmed \
    --llm_model gpt-4o-mini \
    --max_workers 8 \
    --use_mirage \
    --mirage_dataset medqa \
    --chunks_limit 10000 \
    --questions_limit 100 \
    > medqa_pool500_accuracy_test.log 2>&1 &

PID=$!
echo "✅ 测试已启动 (PID: $PID)"
echo "📝 日志文件: medqa_pool500_accuracy_test.log"
echo ""
echo "监控命令:"
echo "  tail -f medqa_pool500_accuracy_test.log | grep -E 'Retrieving:|Accuracy:'"
echo ""
echo "快速查看进度:"
echo "  grep 'Retrieving:' medqa_pool500_accuracy_test.log | tail -5"
echo ""
echo "完成后查看结果:"
echo "  python compare_performance.py"
echo ""
