#!/bin/bash
#
# MedMCQA 测试（使用现有的 50k chunks 图）
# 复用 pubmed_mirage_medqa 的图索引
#

echo "============================================"
echo "MedMCQA Test (使用现有 50k 图)"
echo "Dataset: MIRAGE/rawdata/medmcqa/data/dev.json"
echo "Graph: pubmed_mirage_medqa (50k chunks, 21万实体)"
echo "Start time: $(date)"
echo "============================================"

# ✅ 使用符号链接，让 medmcqa 使用 medqa 的图
if [ ! -d "import/pubmed_mirage_medmcqa" ]; then
    echo "创建符号链接: pubmed_mirage_medmcqa -> pubmed_mirage_medqa"
    ln -s pubmed_mirage_medqa import/pubmed_mirage_medmcqa
    echo "✅ 符号链接已创建"
else
    # 检查是否是符号链接
    if [ -L "import/pubmed_mirage_medmcqa" ]; then
        echo "✅ 符号链接已存在"
    else
        echo "⚠️  import/pubmed_mirage_medmcqa 已存在但不是符号链接"
        echo "   将使用现有目录中的图"
    fi
fi

echo ""
echo "图信息:"
echo "  - Passages: 49,999"
echo "  - Entities: 212,532" 
echo "  - Sentences: 279,428"
echo ""

# ✅ 限制并发数量避免SSH断连
nohup python run.py \
  --use_hf_ner \
  --embedding_model model/all-mpnet-base-v2 \
  --dataset_name pubmed \
  --llm_model gpt-4o-mini \
  --max_workers 4 \
  --use_mirage \
  --mirage_dataset medmcqa \
  --chunks_limit 10000 \
  > medmcqa_50k_$(date +%Y%m%d_%H%M%S).log 2>&1 &

PID=$!
echo "============================================"
echo "Job started in background"
echo "Process ID: $PID"
echo "Monitor with: tail -f medmcqa_50k_*.log"
echo "============================================"

# 保存 PID
echo $PID > medmcqa_50k.pid

echo ""
echo "💡 说明:"
echo "  - 使用现有的 50k chunks 图（不需要重新构建）"
echo "  - 图中包含 21万+ 实体，检索效果更好"
echo "  - 预计启动更快（跳过 NER 处理）"
echo ""
