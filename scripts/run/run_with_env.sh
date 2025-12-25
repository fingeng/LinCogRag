#!/bin/bash

# 设置环境变量
export OPENAI_API_KEY="your-api-key-here"
export OPENAI_BASE_URL="your-base-url-here"

# 激活 conda 环境
source ~/miniconda3/etc/profile.d/conda.sh
conda activate medgraphrag

# 进入项目目录
cd /home/maoxy23/projects/LinearRAG

# 运行程序
python run.py \
    --spacy_model en_core_sci_scibert \
    --embedding_model model/all-mpnet-base-v2 \
    --dataset_name pubmed \
    --llm_model gpt-4o-mini \
    --max_workers 8 \
    --use_mirage \
    --mirage_dataset mmlu \
    --chunks_limit 1000
