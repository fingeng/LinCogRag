#!/bin/bash
#
# 候选集预筛选优化测试
#

cd /home/maoxy23/projects/LinearRAG

echo "=================================================="
echo "候选集预筛选优化 - 测试运行"
echo "=================================================="
echo ""
echo "✅ 已实施的优化:"
echo "   1. 参数优化 (max_iterations=2, threshold=0.3)"
echo "   2. 候选集预筛选 (只在top-200中图搜索)"
echo ""
echo "📊 预期效果:"
echo "   - 速度: 16秒 → 5-8秒/问题 (2-3倍提速)"
echo "   - 总速度提升: 90秒 → 5-8秒 (11-18倍)"
echo "   - 准确率影响: -1% ~ -2%"
echo ""
echo "=================================================="
echo ""

# 停止可能运行的旧进程
OLD_PID=$(pgrep -f "run.py" || true)
if [ -n "$OLD_PID" ]; then
    echo "⚠️  发现运行中的进程: $OLD_PID"
    echo "正在停止..."
    kill $OLD_PID 2>/dev/null || true
    sleep 2
    echo "✅ 已停止"
fi

# 运行测试 (先100个问题)
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
    > medqa_candidate_filtering_100q.log 2>&1 &

PID=$!
echo "✅ 测试已启动 (PID: $PID)"
echo "📝 日志文件: medqa_candidate_filtering_100q.log"
echo ""
echo "监控命令:"
echo "  tail -f medqa_candidate_filtering_100q.log | grep 'Retrieving:'"
echo ""
echo "对比之前的速度:"
echo "  - 原始配置: 90秒/问题"
echo "  - 参数优化后: 16.5秒/问题"
echo "  - 预期现在: 5-8秒/问题"
echo ""
echo "等待约10-15分钟后，可以运行以下命令查看结果:"
echo "  grep 'Retrieving:' medqa_candidate_filtering_100q.log | tail -10"
echo "  grep 'Accuracy:' medqa_candidate_filtering_100q.log"
echo ""
