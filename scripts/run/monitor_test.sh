#!/bin/bash
#
# 实时监控候选集预筛选测试进度
#

LOG_FILE="medqa_candidate_filtering_100q.log"

echo "=================================================="
echo "实时监控 - 候选集预筛选测试"
echo "=================================================="
echo ""

# 检查日志文件是否存在
if [ ! -f "$LOG_FILE" ]; then
    echo "⚠️  日志文件不存在: $LOG_FILE"
    echo "请先运行: ./test_candidate_filtering.sh"
    exit 1
fi

# 实时监控函数
monitor() {
    echo "📊 实时检索进度:"
    echo ""
    
    while true; do
        # 获取最新的检索进度
        LAST_LINE=$(grep "Retrieving:" "$LOG_FILE" | tail -1)
        
        if [ -n "$LAST_LINE" ]; then
            # 提取进度和速度
            PROGRESS=$(echo "$LAST_LINE" | grep -oP '\d+/\d+')
            SPEED=$(echo "$LAST_LINE" | grep -oP '\d+\.\d+s/it' | head -1)
            PERCENT=$(echo "$LAST_LINE" | grep -oP '\d+%' | head -1)
            
            # 清屏并显示
            clear
            echo "=================================================="
            echo "实时监控 - 候选集预筛选测试"
            echo "=================================================="
            echo ""
            echo "📝 日志文件: $LOG_FILE"
            echo ""
            echo "📊 当前进度: $PROGRESS ($PERCENT)"
            echo "⚡ 当前速度: $SPEED"
            echo ""
            
            # 显示最近10条检索记录
            echo "最近检索速度 (最后10个问题):"
            echo "-------------------------------------------"
            grep "Retrieving:" "$LOG_FILE" | tail -10 | while read line; do
                SPEED_ITEM=$(echo "$line" | grep -oP '\d+\.\d+s/it' | head -1)
                PERCENT_ITEM=$(echo "$line" | grep -oP '\d+%' | head -1)
                echo "  $PERCENT_ITEM - $SPEED_ITEM"
            done
            
            # 计算平均速度
            echo ""
            echo "平均速度统计:"
            echo "-------------------------------------------"
            AVG_SPEED=$(grep "Retrieving:" "$LOG_FILE" | grep -oP '\d+\.\d+s/it' | tail -20 | \
                awk '{sum+=$1; count++} END {if(count>0) print sum/count; else print "N/A"}')
            echo "  最近20个问题平均: ${AVG_SPEED} 秒/问题"
            
            # 检查是否完成
            if grep -q "Overall Results" "$LOG_FILE"; then
                echo ""
                echo "=================================================="
                echo "✅ 测试已完成!"
                echo "=================================================="
                echo ""
                
                # 显示结果
                grep "LLM Accuracy:" "$LOG_FILE"
                grep "Contain Accuracy:" "$LOG_FILE"
                
                echo ""
                echo "运行性能对比分析:"
                echo "  python compare_performance.py"
                echo ""
                break
            fi
            
            echo ""
            echo "按 Ctrl+C 停止监控"
            echo "=================================================="
        else
            echo "⏳ 等待测试开始..."
        fi
        
        sleep 5
    done
}

# 提供选择
echo "选择监控模式:"
echo "  1. 实时监控 (每5秒刷新)"
echo "  2. 查看最近10条"
echo "  3. 查看完整日志"
echo ""
read -p "请选择 (1-3): " choice

case $choice in
    1)
        monitor
        ;;
    2)
        echo ""
        echo "最近10条检索记录:"
        echo "=================================================="
        grep "Retrieving:" "$LOG_FILE" | tail -10
        echo ""
        
        # 计算平均速度
        AVG=$(grep "Retrieving:" "$LOG_FILE" | grep -oP '\d+\.\d+s/it' | tail -10 | \
            awk '{sum+=$1; count++} END {print sum/count}')
        echo "平均速度: $AVG 秒/问题"
        ;;
    3)
        tail -f "$LOG_FILE"
        ;;
    *)
        echo "无效选择"
        ;;
esac
