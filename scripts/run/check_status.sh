#!/bin/bash
#
# 快速检查测试状态
#

LOG_FILE="medqa_pool500_accuracy_test.log"

echo "=================================================="
echo "测试状态检查"
echo "=================================================="
echo ""

# 检查文件是否存在
if [ ! -f "$LOG_FILE" ]; then
    echo "⚠️  日志文件不存在: $LOG_FILE"
    exit 1
fi

# 检查是否有错误
if grep -q "Error\|Traceback\|Exception" "$LOG_FILE"; then
    echo "❌ 发现错误:"
    grep -A 5 "Error\|Traceback\|Exception" "$LOG_FILE" | head -20
    exit 1
fi

# 检查是否已开始检索
if ! grep -q "Retrieving:" "$LOG_FILE"; then
    echo "⏳ 正在初始化..."
    echo ""
    echo "最新日志:"
    tail -10 "$LOG_FILE"
    exit 0
fi

# 显示检索进度
echo "📊 检索进度:"
LAST_LINE=$(grep "Retrieving:" "$LOG_FILE" | tail -1)
echo "  $LAST_LINE"
echo ""

# 计算平均速度
echo "⚡ 速度统计:"
AVG_SPEED=$(grep "Retrieving:" "$LOG_FILE" | grep -oP '\d+\.\d+s/it' | tail -10 | \
    awk '{sum+=$1; count++} END {if(count>0) print sum/count; else print "N/A"}')
echo "  最近10个问题平均: ${AVG_SPEED} 秒/问题"

# 提取最近5条
echo ""
echo "📝 最近5条记录:"
grep "Retrieving:" "$LOG_FILE" | tail -5 | while read line; do
    PERCENT=$(echo "$line" | grep -oP '\d+%' | head -1)
    SPEED=$(echo "$line" | grep -oP '\d+\.\d+s/it' | head -1)
    echo "  $PERCENT - $SPEED"
done

# 检查是否完成
if grep -q "Overall Results" "$LOG_FILE"; then
    echo ""
    echo "=================================================="
    echo "✅ 测试已完成!"
    echo "=================================================="
    echo ""
    
    # 显示结果
    echo "📊 最终结果:"
    grep "LLM Accuracy:" "$LOG_FILE"
    grep "Contain Accuracy:" "$LOG_FILE"
    echo ""
    
    echo "运行完整对比分析:"
    echo "  python complete_comparison.py"
else
    echo ""
    echo "状态: 🔄 运行中"
    
    # 估算剩余时间
    CURRENT=$(grep "Retrieving:" "$LOG_FILE" | tail -1 | grep -oP '\d+/\d+' | cut -d'/' -f1)
    TOTAL=$(grep "Retrieving:" "$LOG_FILE" | tail -1 | grep -oP '\d+/\d+' | cut -d'/' -f2)
    
    if [ -n "$CURRENT" ] && [ -n "$TOTAL" ] && [ "$AVG_SPEED" != "N/A" ]; then
        REMAINING=$((TOTAL - CURRENT))
        TIME_LEFT=$(echo "$REMAINING * $AVG_SPEED" | bc)
        MINUTES=$(echo "$TIME_LEFT / 60" | bc)
        echo "预计剩余时间: 约 $MINUTES 分钟"
    fi
fi

echo ""
