#!/bin/bash

# Run LinearRAG on all 5 MIRAGE benchmarks
# Using PubMed full dataset (23.9M passages)

cd /home/maoxy23/projects/LinearRAG

export CUDA_VISIBLE_DEVICES=0,6,7
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

BENCHMARKS=("mmlu" "medqa" "pubmedqa" "medmcqa" "mmlu_medical")

for BENCHMARK in "${BENCHMARKS[@]}"; do
    echo "========================================"
    echo "Running benchmark: $BENCHMARK"
    echo "========================================"
    
    LOG_FILE="results/pubmed_full_${BENCHMARK}.log"
    mkdir -p results
    
    python run.py \
      --use_hf_ner \
      --embedding_model model/all-mpnet-base-v2 \
      --dataset_name pubmed \
      --llm_model gpt-4o-mini \
      --max_workers 2 \
      --use_mirage \
      --mirage_dataset $BENCHMARK \
      > $LOG_FILE 2>&1
    
    echo "✅ Completed: $BENCHMARK"
    echo "Log saved to: $LOG_FILE"
    echo ""
done

echo "========================================"
echo "All benchmarks completed!"
echo "========================================"
