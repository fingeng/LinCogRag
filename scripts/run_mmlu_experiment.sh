#!/bin/bash

echo "========================================================================"
echo "MMLU Medical QA Experiment with Biomedical NER"
echo "========================================================================"

# Configuration
CHUNKS_LIMIT=100
MAX_WORKERS=4
DATASET_NAME="pubmed"
LLM_MODEL="gpt-4o-mini"

echo ""
echo "Configuration:"
echo "  NER Model: biomedical-ner-all (local)"
echo "  Chunks: $CHUNKS_LIMIT"
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
echo "🚀 Starting MMLU experiment..."
echo ""

python run.py \
  --use_hf_ner \
  --embedding_model model/all-mpnet-base-v2 \
  --dataset_name $DATASET_NAME \
  --llm_model $LLM_MODEL \
  --max_workers $MAX_WORKERS \
  --use_mirage \
  --mirage_dataset mmlu \
  --chunks_limit $CHUNKS_LIMIT

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
    echo "Next steps:"
    echo "  1. Check predictions: cat results/pubmed_mirage_mmlu/predictions.json | jq '.[] | {question: .question, pred: .pred_answer, gold: .gold_answer}' | head -20"
    echo "  2. Analyze graph: python scripts/analyze_graph.py"
    echo "  3. View diagnostics: python scripts/diagnose_graph.py"
    echo ""
else
    echo ""
    echo "========================================================================"
    echo "❌ Experiment failed!"
    echo "========================================================================"
    echo ""
    echo "Check logs above for errors"
fi
