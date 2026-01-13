#!/bin/bash
# LinCog-RAG Benchmark Runner
# Usage: bash experiments/run_benchmark.sh

# Set working directory
cd "$(dirname "$0")/.."

echo "=============================================="
echo "LinCog-RAG Benchmark"
echo "=============================================="
echo "Working directory: $(pwd)"
echo ""

# Set OpenAI API configuration
# 请设置环境变量 OPENAI_API_KEY 和 OPENAI_BASE_URL (可选)
# 例如: export OPENAI_API_KEY="your-api-key"
if [ -z "$OPENAI_API_KEY" ]; then
    echo "❌ Error: OPENAI_API_KEY environment variable is not set"
    echo "Please run: export OPENAI_API_KEY=\"your-api-key\""
    exit 1
fi

# 如果未设置 BASE_URL，使用默认值
export OPENAI_BASE_URL="${OPENAI_BASE_URL:-https://api.openai.com/v1}"

echo "API Configuration:"
echo "  OPENAI_BASE_URL: $OPENAI_BASE_URL"
echo "  OPENAI_API_KEY: ${OPENAI_API_KEY:0:10}***"
echo ""

# Check if CUDA is available
python3 -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
echo ""

# Run benchmark
echo "Starting benchmark..."
python3 experiments/run_lincog_benchmark.py 2>&1 | tee artifacts/lincog_benchmark/benchmark_$(date +%Y%m%d_%H%M%S).log

echo ""
echo "Benchmark completed!"











