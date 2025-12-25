#!/bin/bash
#
# 监控 MedMCQA 测试进度
#

# 获取最新的日志文件
LOG_FILE=$(ls -t medmcqa_full_*.log 2>/dev/null | head -1)
PID_FILE="medmcqa_test.pid"

if [ ! -f "$PID_FILE" ]; then
    echo "❌ 找不到 PID 文件: $PID_FILE"
    echo "请先运行: ./run_medmcqa_full.sh"
    exit 1
fi

PID=$(cat "$PID_FILE")

echo "=================================================="
echo "MedMCQA 完整测试监控"
echo "=================================================="
echo ""
echo "进程ID: $PID"
echo "日志文件: $LOG_FILE"
echo "检查时间: $(date)"
echo ""

# 检查进程是否运行
if ! ps -p $PID > /dev/null 2>&1; then
    echo "❌ 进程未运行 (PID: $PID)"
    echo ""
    if [ -f "$LOG_FILE" ]; then
        echo "检查最终结果:"
        if grep -q "Overall Results" "$LOG_FILE"; then
            echo "✅ 测试已完成!"
            echo ""
            grep -E "LLM Accuracy:|Contain Accuracy:" "$LOG_FILE" | grep -v "sample" | tail -2
        else
            echo "进程已终止，检查错误:"
            tail -50 "$LOG_FILE" | grep -i "error\|exception\|traceback" || echo "未发现明显错误"
        fi
    fi
    exit 1
fi

echo "✅ 进程运行中"
echo ""

# 检查日志文件
if [ ! -f "$LOG_FILE" ]; then
    echo "⚠️  日志文件未找到"
    exit 0
fi

FILE_SIZE=$(stat -c%s "$LOG_FILE" 2>/dev/null || echo "0")

# 如果文件太小，可能还在初始化
if [ "$FILE_SIZE" -lt 500 ]; then
    echo "⏳ 正在初始化..."
    echo "   当前日志大小: $FILE_SIZE 字节"
    exit 0
fi

# 查找最新的检索进度
echo "📊 最新进度:"
LAST_LINE=$(grep "Retrieving:" "$LOG_FILE" 2>/dev/null | tail -1)

if [ -z "$LAST_LINE" ]; then
    echo "  ⏳ 正在初始化..."
    echo ""
    echo "最新日志:"
    tail -10 "$LOG_FILE"
else
    echo "$LAST_LINE"
    echo ""
    
    # 提取进度数字
    PROGRESS=$(echo "$LAST_LINE" | grep -oP '\d+/\d+' | head -1)
    if [ -n "$PROGRESS" ]; then
        CURRENT=$(echo "$PROGRESS" | cut -d'/' -f1)
        TOTAL=$(echo "$PROGRESS" | cut -d'/' -f2)
        PERCENT=$(echo "scale=1; $CURRENT * 100 / $TOTAL" | bc 2>/dev/null || echo "?")
        echo "📈 进度: $CURRENT / $TOTAL ($PERCENT%)"
    fi
    
    # 计算平均速度
    echo ""
    echo "⚡ 速度统计:"
    RECENT_SPEEDS=$(grep "Retrieving:" "$LOG_FILE" | tail -50 | grep -oP '\d+\.\d+(?=s/it)')
    
    if [ -n "$RECENT_SPEEDS" ]; then
        AVG_SPEED=$(echo "$RECENT_SPEEDS" | awk '{sum+=$1; count++} END {if(count>0) printf "%.2f", sum/count}')
        echo "  平均: ${AVG_SPEED}s/问题"
        
        # 估算剩余时间
        if [ -n "$CURRENT" ] && [ -n "$TOTAL" ] && [ "$AVG_SPEED" != "" ]; then
            REMAINING=$((TOTAL - CURRENT))
            TIME_LEFT_MIN=$(echo "$REMAINING * $AVG_SPEED / 60" | bc 2>/dev/null)
            echo "  预计剩余: 约 $TIME_LEFT_MIN 分钟"
        fi
    fi
fi

echo ""

# 检查是否完成
if grep -q "Overall Results" "$LOG_FILE"; then
    echo "=================================================="
    echo "✅ 测试已完成!"
    echo "=================================================="
    echo ""
    echo "📊 最终结果:"
    grep -E "LLM Accuracy:|Contain Accuracy:" "$LOG_FILE" | grep -v "sample" | tail -2
    echo ""
else
    echo "状态: 🔄 运行中"
fi

echo ""
echo "💡 提示:"
echo "  - 持续监控: watch -n 30 './monitor_medmcqa.sh'"
echo "  - 查看日志: tail -f $LOG_FILE"
echo "  - 终止测试: kill $PID"
echo ""
