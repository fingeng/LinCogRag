#!/bin/bash
#
# BioASQ快速测试 - 使用已有的50k图
#

# 激活环境
conda activate medgraphrag

# API配置：请先设置环境变量
if [ -z "$OPENAI_API_KEY" ]; then
    echo "❌ Error: OPENAI_API_KEY not set. Run: export OPENAI_API_KEY=\"your-key\""
    exit 1
fi
export OPENAI_BASE_URL="${OPENAI_BASE_URL:-https://api.openai.com/v1}"

cd /home/maoxy23/projects/LinearRAG

echo "============================================"
echo "BioASQ Quick Test (3 questions)"
echo "使用已构建的50k图: import/pubmed_mirage_medqa"
echo "Start time: $(date)"
echo "============================================"
echo ""

# 创建符号链接,让bioasq使用medqa的图
MEDQA_GRAPH="import/pubmed_mirage_medqa"
BIOASQ_GRAPH="import/pubmed_mirage_bioasq"

if [ ! -d "$BIOASQ_GRAPH" ]; then
    echo "创建符号链接: $BIOASQ_GRAPH -> $MEDQA_GRAPH"
    ln -s "$(pwd)/$MEDQA_GRAPH" "$BIOASQ_GRAPH"
fi

# 运行测试
python run.py \
  --use_hf_ner \
  --embedding_model model/all-mpnet-base-v2 \
  --dataset_name pubmed \
  --llm_model gpt-4o-mini \
  --max_workers 4 \
  --use_mirage \
  --mirage_dataset bioasq \
  --questions_limit 3 \
  --chunks_limit 50000

echo ""
echo "============================================"
echo "End time: $(date)"
echo "============================================"
