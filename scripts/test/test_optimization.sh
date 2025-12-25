#!/bin/bash
#
# LinearRAG 医疗领域优化 - 快速测试脚本
# 用于对比不同配置的性能
#

set -e

PROJECT_DIR="/home/maoxy23/projects/LinearRAG"
cd "$PROJECT_DIR"

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 打印函数
print_header() {
    echo -e "\n${BLUE}========================================${NC}"
    echo -e "${BLUE}$1${NC}"
    echo -e "${BLUE}========================================${NC}\n"
}

print_success() {
    echo -e "${GREEN}✅ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

print_error() {
    echo -e "${RED}❌ $1${NC}"
}

# 停止当前运行
stop_current_run() {
    print_header "停止当前运行"
    
    PID=$(pgrep -f "python run.py" || true)
    if [ -n "$PID" ]; then
        print_warning "发现运行中的进程: PID=$PID"
        echo "是否停止? (y/n)"
        read -r response
        if [[ "$response" =~ ^[Yy]$ ]]; then
            kill "$PID"
            print_success "已停止进程 $PID"
            sleep 2
        else
            print_error "取消操作"
            exit 1
        fi
    else
        print_success "没有运行中的进程"
    fi
}

# 备份当前配置
backup_config() {
    print_header "备份配置文件"
    
    if [ -f "src/config.py" ]; then
        cp src/config.py src/config_backup_$(date +%Y%m%d_%H%M%S).py
        print_success "配置已备份"
    fi
}

# 生成优化配置
generate_optimized_config() {
    local config_type=$1
    
    print_header "生成 $config_type 配置"
    
    case $config_type in
        "quick_fix")
            MAX_ITER=2
            ITER_THRESH=0.3
            TOP_K_SENT=5
            DESC="快速优化 (推荐)"
            ;;
        "balanced")
            MAX_ITER=2
            ITER_THRESH=0.25
            TOP_K_SENT=8
            DESC="平衡配置"
            ;;
        "aggressive")
            MAX_ITER=1
            ITER_THRESH=0.5
            TOP_K_SENT=10
            DESC="激进优化"
            ;;
        *)
            print_error "未知配置类型: $config_type"
            exit 1
            ;;
    esac
    
    cat > src/config.py << EOF
from dataclasses import dataclass
from src.utils import LLM_Model

@dataclass
class LinearRAGConfig:
    """
    LinearRAG 配置 - $DESC
    生成时间: $(date '+%Y-%m-%d %H:%M:%S')
    优化参数:
        - max_iterations: $MAX_ITER
        - iteration_threshold: $ITER_THRESH
        - top_k_sentence: $TOP_K_SENT
    """
    def __init__(
        self,
        embedding_model,
        dataset_name,
        spacy_model="en_ner_bc5cdr_md",
        max_workers=8,
        llm_model=None,
        use_hf_ner=True,
        use_enhanced_ner=True,
        working_dir="import",
        batch_size=32,
        retrieval_top_k=32,
        max_iterations=$MAX_ITER,  # ✅ 优化
        iteration_threshold=$ITER_THRESH,  # ✅ 优化
        top_k_sentence=$TOP_K_SENT,  # ✅ 优化
        passage_ratio=0.7,
        passage_node_weight=1.0,
        damping=0.85,
    ):
        # Model parameters
        self.embedding_model = embedding_model
        self.spacy_model = spacy_model
        self.llm_model = llm_model
        
        # NER strategy
        self.use_hf_ner = use_hf_ner
        self.use_enhanced_ner = use_enhanced_ner
        
        # Dataset parameters
        self.dataset_name = dataset_name
        self.working_dir = working_dir
        
        # Processing parameters
        self.max_workers = max_workers
        self.batch_size = batch_size
        
        # Retrieval parameters
        self.retrieval_top_k = retrieval_top_k
        self.max_iterations = max_iterations
        self.iteration_threshold = iteration_threshold
        self.top_k_sentence = top_k_sentence
        
        # Graph parameters
        self.passage_ratio = passage_ratio
        self.passage_node_weight = passage_node_weight
        self.damping = damping
EOF
    
    print_success "已生成 $DESC 配置"
    echo "   max_iterations: $MAX_ITER"
    echo "   iteration_threshold: $ITER_THRESH"
    echo "   top_k_sentence: $TOP_K_SENT"
}

# 运行测试
run_test() {
    local config_name=$1
    local num_questions=$2
    
    print_header "运行测试: $config_name ($num_questions 个问题)"
    
    LOG_FILE="test_${config_name}_${num_questions}q.log"
    
    python run.py \
        --use_hf_ner \
        --embedding_model model/all-mpnet-base-v2 \
        --dataset_name pubmed \
        --llm_model gpt-4o-mini \
        --max_workers 8 \
        --use_mirage \
        --mirage_dataset medqa \
        --chunks_limit 10000 \
        --questions_limit "$num_questions" \
        > "$LOG_FILE" 2>&1 &
    
    PID=$!
    print_success "测试已启动: PID=$PID"
    print_success "日志文件: $LOG_FILE"
    
    echo ""
    echo "监控进度:"
    echo "  tail -f $LOG_FILE | grep -E 'Retrieving:|Accuracy:'"
    echo ""
    echo "查看速度:"
    echo "  grep 'Retrieving:' $LOG_FILE | tail -5"
    echo ""
}

# 分析结果
analyze_results() {
    print_header "分析测试结果"
    
    echo -e "${BLUE}文件名                     问题数  平均速度      准确率${NC}"
    echo "----------------------------------------------------------------"
    
    for log_file in test_*.log; do
        if [ -f "$log_file" ]; then
            # 提取配置名
            config_name=$(echo "$log_file" | sed 's/test_\(.*\)_.*q\.log/\1/')
            
            # 提取问题数
            questions=$(grep -oP "Total questions loaded: \K\d+" "$log_file" | head -1)
            
            # 提取平均速度 (从最后几行的进度)
            avg_time=$(grep "Retrieving:" "$log_file" | tail -5 | grep -oP "\d+\.\d+s/it" | tail -1 || echo "N/A")
            
            # 提取准确率
            accuracy=$(grep -oP "Accuracy: \K[\d.]+%" "$log_file" | tail -1 || echo "运行中")
            
            printf "%-25s %-8s %-14s %s\n" "$log_file" "$questions" "$avg_time" "$accuracy"
        fi
    done
    
    echo ""
}

# 主菜单
main_menu() {
    print_header "LinearRAG 医疗领域优化测试"
    
    echo "请选择操作:"
    echo ""
    echo "  1. 停止当前运行"
    echo "  2. 快速测试 (100个问题，快速优化配置)"
    echo "  3. 完整测试 (全部问题，快速优化配置)"
    echo "  4. 对比测试 (多种配置，各100个问题)"
    echo "  5. 分析现有结果"
    echo "  6. 恢复原始配置"
    echo "  0. 退出"
    echo ""
    echo -n "请输入选择: "
    read -r choice
    
    case $choice in
        1)
            stop_current_run
            ;;
        2)
            backup_config
            generate_optimized_config "quick_fix"
            run_test "quick_fix" 100
            ;;
        3)
            backup_config
            generate_optimized_config "quick_fix"
            run_test "quick_fix" 1273
            ;;
        4)
            print_header "对比测试模式"
            backup_config
            
            # 测试多种配置
            for config in "quick_fix" "balanced" "aggressive"; do
                generate_optimized_config "$config"
                run_test "$config" 100
                sleep 5  # 等待启动
            done
            
            print_success "所有测试已启动"
            print_warning "请等待测试完成后，运行选项5分析结果"
            ;;
        5)
            analyze_results
            ;;
        6)
            print_header "恢复原始配置"
            
            BACKUP=$(ls -t src/config_backup_*.py 2>/dev/null | head -1)
            if [ -n "$BACKUP" ]; then
                cp "$BACKUP" src/config.py
                print_success "已恢复配置: $BACKUP"
            else
                print_error "未找到备份文件"
            fi
            ;;
        0)
            print_success "退出"
            exit 0
            ;;
        *)
            print_error "无效选择"
            ;;
    esac
}

# 运行主菜单
if [ $# -eq 0 ]; then
    main_menu
else
    # 命令行模式
    case $1 in
        stop)
            stop_current_run
            ;;
        quick)
            backup_config
            generate_optimized_config "quick_fix"
            run_test "quick_fix" "${2:-100}"
            ;;
        compare)
            backup_config
            for config in "quick_fix" "balanced" "aggressive"; do
                generate_optimized_config "$config"
                run_test "$config" 100
                sleep 5
            done
            ;;
        analyze)
            analyze_results
            ;;
        *)
            echo "用法: $0 [stop|quick|compare|analyze]"
            exit 1
            ;;
    esac
fi
