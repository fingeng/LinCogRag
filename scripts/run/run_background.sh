#!/bin/bash

# LinearRAG 后台运行脚本
cd /home/maoxy23/projects/LinearRAG

echo "Starting LinearRAG with MIRAGE benchmark..."
echo "Start time: $(date)"

python run.py \
    --spacy_model en_core_sci_scibert \
    --embedding_model model/all-mpnet-base-v2 \
    --dataset_name pubmed \
    --llm_model gpt-4o-mini \
    --max_workers 16 \
    --use_mirage \
    --mirage_dataset mmlu

echo "End time: $(date)"
echo "Process completed!"
