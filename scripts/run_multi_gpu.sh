#!/bin/bash

echo "========================================================================"
echo "Multi-GPU LinearRAG Experiment"
echo "========================================================================"

# 1. 检查当前任务
echo "Step 1: Checking for existing processes..."
pgrep -f "python run.py" && {
    echo "⚠️  Found existing python run.py process"
    read -p "Kill it? (y/n): " answer
    if [ "$answer" = "y" ]; then
        pkill -f "python run.py"
        echo "✅ Killed"
        sleep 2
    fi
}

# 2. 设置 GPU
export CUDA_VISIBLE_DEVICES=0,6,7
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo ""
echo "Step 2: GPU Configuration"
echo "   CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
echo "   Available GPUs:"
nvidia-smi --query-gpu=index,name,memory.free --format=csv,noheader | head -3

# 3. 清理旧数据(可选)
echo ""
read -p "Step 3: Clear old index data? (y/n): " clear_data
if [ "$clear_data" = "y" ]; then
    echo "Clearing import/pubmed_mirage_mmlu/..."
    rm -rf import/pubmed_mirage_mmlu/
    echo "✅ Cleared"
fi

# 4. 测试模式选择
echo ""
echo "Step 4: Choose dataset size"
echo "   1) Test (10k passages, ~2 hours)"
echo "   2) Medium (100k passages, ~8 hours)"
echo "   3) Full (all ~24M passages, ~26 hours)"
read -p "Select (1/2/3): " size_choice

case $size_choice in
    1)
        CHUNKS_LIMIT="--chunks_limit 500"
        LOG_FILE="job_3gpu_test.log"
        echo "Selected: Test mode (500 chunks)"
        ;;
    2)
        CHUNKS_LIMIT="--chunks_limit 5000"
        LOG_FILE="job_3gpu_medium.log"
        echo "Selected: Medium mode (5000 chunks)"
        ;;
    3)
        CHUNKS_LIMIT=""
        LOG_FILE="job_3gpu_full.log"
        echo "Selected: Full dataset"
        ;;
    *)
        echo "Invalid choice, using test mode"
        CHUNKS_LIMIT="--chunks_limit 500"
        LOG_FILE="job_3gpu_test.log"
        ;;
esac

# 5. 启动任务
echo ""
echo "Step 5: Starting experiment"
echo "   Log file: $LOG_FILE"
echo "   Command:"
echo "     python run.py \\"
echo "       --use_hf_ner \\"
echo "       --embedding_model model/all-mpnet-base-v2 \\"
echo "       --dataset_name pubmed \\"
echo "       --llm_model gpt-4o-mini \\"
echo "       --max_workers 2 \\"
echo "       --use_mirage \\"
echo "       --mirage_dataset mmlu \\"
echo "       $CHUNKS_LIMIT"

read -p "Press Enter to start..."

nohup python run.py \
  --use_hf_ner \
  --embedding_model model/all-mpnet-base-v2 \
  --dataset_name pubmed \
  --llm_model gpt-4o-mini \
  --max_workers 2 \
  --use_mirage \
  --mirage_dataset mmlu \
  $CHUNKS_LIMIT \
  > $LOG_FILE 2>&1 &

PID=$!
echo ""
echo "========================================================================"
echo "✅ Started! PID: $PID"
echo "========================================================================"
echo ""
echo "Monitor with:"
echo "   tail -f $LOG_FILE"
echo "   watch -n 2 nvidia-smi"
echo ""
echo "Check GPU usage:"
echo "   nvidia-smi"
echo ""
echo "Stop with:"
echo "   kill $PID"
echo ""
