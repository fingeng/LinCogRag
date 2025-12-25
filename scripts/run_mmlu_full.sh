#!/bin/bash

echo "========================================================================"
echo "MMLU Full Dataset Experiment with Biomedical NER"
echo "========================================================================"

# Configuration
MAX_WORKERS=2  # 🔧 降低并发数以节省内存
DATASET_NAME="pubmed"
LLM_MODEL="gpt-4o-mini"

echo ""
echo "Configuration:"
echo "  NER Model: biomedical-ner-all (local)"
echo "  Chunks: ALL (1166 JSONL files, ~100k+ passages)"  # 🔧 使用全部数据
echo "  Workers: $MAX_WORKERS"
echo "  LLM: $LLM_MODEL"
echo "  Dataset: MMLU Medical"
echo ""

# Clean previous data
echo "🧹 Cleaning previous experiment data..."
if [ -d "import/pubmed_mirage_mmlu" ]; then
    backup_dir="import/pubmed_mirage_mmlu_backup_$(date +%Y%m%d_%H%M%S)"
    echo "   Backing up to: $backup_dir"
    mv import/pubmed_mirage_mmlu "$backup_dir"
fi

# Set memory optimization
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Run experiment
echo ""
echo "🚀 Starting MMLU FULL experiment..."
echo "⚠️  This will take several hours to complete!"
echo ""

python run.py \
  --use_hf_ner \
  --embedding_model model/all-mpnet-base-v2 \
  --dataset_name $DATASET_NAME \
  --llm_model $LLM_MODEL \
  --max_workers $MAX_WORKERS \
  --use_mirage \
  --mirage_dataset mmlu
  # 🔧 注意：移除了 --chunks_limit 参数

# Check results
if [ $? -eq 0 ]; then
    echo ""
    echo "========================================================================"
    echo "✅ Experiment completed successfully!"
    echo "========================================================================"
    echo ""
    echo "Results location:"
    echo "  Predictions: results/pubmed_mirage_mmlu/predictions.json"
    echo "  Graph: import/pubmed_mirage_mmlu/LinearRAG.graphml"
    echo "  NER results: import/pubmed_mirage_mmlu/ner_results.json"
    echo ""
else
    echo ""
    echo "========================================================================"
    echo "❌ Experiment failed!"
    echo "========================================================================"
fi
