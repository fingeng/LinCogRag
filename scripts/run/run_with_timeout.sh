#!/bin/bash

# 设置环境变量
export OPENAI_API_KEY="your-key"
export OPENAI_BASE_URL="your-url"

# 激活环境
source ~/miniconda3/etc/profile.d/conda.sh
conda activate medgraphrag

cd /home/maoxy23/projects/LinearRAG

# 使用 timeout 命令,10分钟没输出就报警
timeout 600 python run.py \
    --spacy_model en_core_sci_scibert \
    --embedding_model model/all-mpnet-base-v2 \
    --dataset_name pubmed \
    --llm_model gpt-4o-mini \
    --max_workers 8 \
    --use_mirage \
    --mirage_dataset mmlu \
    --chunks_limit 1000

if [ $? -eq 124 ]; then
    echo "⚠️  Process timeout after 10 minutes"
fi
