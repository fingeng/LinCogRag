#!/bin/bash
#
# 监控完整MedQA测试进度
#

PID=3562611
LOG_FILE="medqa_full_20251201_145611.log"

echo "=================================================="
echo "MedQA 完整测试监控 (1273问题)"
echo "=================================================="
echo ""
echo "进程ID: $PID"
echo "日志文件: $LOG_FILE"
echo ""

# 检查进程是否运行
if ! ps -p $PID > /dev/null 2>&1; then
    echo "❌ 进程未运行"
    exit 1
fi

echo "✅ 进程运行中"
echo ""

# 检查日志文件
if [ ! -f "$LOG_FILE" ]; then
    echo "⚠️  日志文件未找到，可能正在初始化..."
    echo ""
    echo "请稍后重试或检查其他日志文件:"
    ls -lht medqa_full_*.log | head -3
    exit 0
fi

# 检查文件大小
FILE_SIZE=$(stat -c%s "$LOG_FILE" 2>/dev/null || echo "0")
if [ "$FILE_SIZE" -lt 100 ]; then
    echo "⏳ 正在初始化加载索引..."
    echo "   当前日志大小: $FILE_SIZE 字节"
    echo "   预计需要1-2分钟加载图数据..."
    exit 0
fi

echo "📊 检索进度:"
# 查找最新的Retrieving行
LAST_LINE=$(grep "Retrieving:" "$LOG_FILE" 2>/dev/null | tail -1)
if [ -z "$LAST_LINE" ]; then
    echo "  ⏳ 正在初始化..."
    echo ""
    echo "最新日志（最后10行）:"
    tail -10 "$LOG_FILE"
else
    echo "  $LAST_LINE"
    echo ""
    
    # 提取进度
    CURRENT=$(echo "$LAST_LINE" | grep -oP '\d+/\d+' | cut -d'/' -f1)
    TOTAL=$(echo "$LAST_LINE" | grep -oP '\d+/\d+' | cut -d'/' -f2)
    
    if [ -n "$CURRENT" ] && [ -n "$TOTAL" ]; then
        PERCENT=$(echo "scale=1; $CURRENT * 100 / $TOTAL" | bc)
        echo "  进度: $CURRENT / $TOTAL ($PERCENT%)"
        
        # 计算平均速度
        echo ""
        echo "⚡ 速度统计:"
        AVG_SPEED=$(grep "Retrieving:" "$LOG_FILE" | tail -50 | \
            grep -oP '\d+\.\d+s/it' | \
            awk '{sum+=$1; count++} END {if(count>0) printf "%.2f", sum/count; else print "N/A"}')
        echo "  最近平均: ${AVG_SPEED} 秒/问题"
        
        # 估算剩余时间
        if [ "$AVG_SPEED" != "N/A" ]; then
            REMAINING=$((TOTAL - CURRENT))
            TIME_LEFT=$(echo "$REMAINING * $AVG_SPEED / 60" | bc)
            echo "  预计剩余: 约 $TIME_LEFT 分钟"
        fi
    fi
fi

# 检查是否完成
echo ""
if grep -q "Overall Results" "$LOG_FILE"; then
    echo "=================================================="
    echo "✅ 测试已完成!"
    echo "=================================================="
    echo ""
    grep -A 5 "Overall Results" "$LOG_FILE"
else
    echo "状态: 🔄 运行中"
    echo ""
    echo "实时监控命令:"
    echo "  watch -n 10 './monitor_full_test.sh'"
    echo "  tail -f $LOG_FILE"
fi

echo ""
