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
export OPENAI_BASE_URL="https://api.chatanywhere.tech"
export OPENAI_API_KEY="sk-RXbQMpzfo7ERxebnz9PTFQruIbAuBQ6odYPnrzaclBmG2vDc"

echo "API Configuration:"
echo "  OPENAI_BASE_URL: $OPENAI_BASE_URL"
echo "  OPENAI_API_KEY: ${OPENAI_API_KEY:0:20}..."
echo ""

# Check if CUDA is available
python3 -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
echo ""

# Run benchmark
echo "Starting benchmark..."
python3 experiments/run_lincog_benchmark.py 2>&1 | tee artifacts/lincog_benchmark/benchmark_$(date +%Y%m%d_%H%M%S).log

echo ""
echo "Benchmark completed!"










