#!/usr/bin/env python3
"""
增量方式：将PubMedQA的500个CONTEXTS chunks添加到已有的图中
这样可以快速验证，而不需要重建整个50k的图
"""

import json
import os
import pickle
import argparse
from pathlib import Path

def load_existing_graph_info(graph_dir='import/pubmed_mirage_medqa'):
    """检查现有图的信息"""
    
    print("="*80)
    print("Checking Existing Graph")
    print("="*80)
    
    if not os.path.exists(graph_dir):
        print(f"❌ Graph directory not found: {graph_dir}")
        return None
    
    # 检查文件
    files = {
        'ner_results': 'ner_results.json',
        'passage_embedding': 'passage_embedding.parquet',
        'sentence_embedding': 'sentence_embedding.parquet',
        'entity_embedding': 'entity_embedding.parquet',
        'graphml': 'LinearRAG.graphml',
    }
    
    info = {'dir': graph_dir, 'files': {}}
    
    for name, filename in files.items():
        filepath = os.path.join(graph_dir, filename)
        if os.path.exists(filepath):
            size = os.path.getsize(filepath)
            info['files'][name] = {
                'path': filepath,
                'size': size,
                'size_mb': size / (1024*1024)
            }
            print(f"✅ {filename}: {size/(1024*1024):.1f} MB")
        else:
            print(f"❌ {filename}: Not found")
            info['files'][name] = None
    
    # 读取NER结果检查passage数量
    if info['files'].get('ner_results'):
        with open(info['files']['ner_results']['path'], 'r') as f:
            ner_data = json.load(f)
        info['num_passages'] = len(ner_data)
        print(f"\n📊 Current passages in graph: {info['num_passages']}")
    
    return info

def load_pubmedqa_chunks(chunks_file='pubmedqa_contexts_chunks.jsonl'):
    """加载PubMedQA chunks"""
    
    print("\n" + "="*80)
    print("Loading PubMedQA Chunks")
    print("="*80)
    
    if not os.path.exists(chunks_file):
        print(f"❌ Chunks file not found: {chunks_file}")
        print(f"   Please run: python extract_pubmedqa_contexts.py")
        return None
    
    chunks = []
    with open(chunks_file, 'r') as f:
        for line in f:
            chunks.append(json.loads(line))
    
    print(f"✅ Loaded {len(chunks)} PubMedQA chunks")
    print(f"   Average length: {sum(len(c['text']) for c in chunks)/len(chunks):.0f} chars")
    
    return chunks

def create_augmented_corpus(original_corpus_files, pubmedqa_chunks, output_file):
    """
    创建增强的corpus：原始50k + PubMedQA 500
    但不实际合并大文件，而是创建一个指向文件的列表
    """
    
    print("\n" + "="*80)
    print("Creating Augmented Corpus Configuration")
    print("="*80)
    
    config = {
        'original_corpus': original_corpus_files,
        'pubmedqa_chunks_file': 'pubmedqa_contexts_chunks.jsonl',
        'total_original': 50000,  # 估计
        'total_pubmedqa': len(pubmedqa_chunks),
        'total': 50000 + len(pubmedqa_chunks),
    }
    
    # 保存配置
    config_file = 'corpus_augmented_config.json'
    with open(config_file, 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"✅ Created augmented corpus configuration")
    print(f"   Original corpus: ~{config['total_original']} passages")
    print(f"   PubMedQA chunks: {config['total_pubmedqa']} passages")
    print(f"   Total: {config['total']} passages")
    print(f"   Config saved to: {config_file}")
    
    return config

def create_quick_test_script(output_file='test_pubmedqa_with_contexts.sh'):
    """
    创建快速测试脚本
    先用500个PubMedQA chunks构建小图测试
    """
    
    script = """#!/bin/bash

# 快速测试：只用500个PubMedQA CONTEXTS构建图

echo "============================================================"
echo "Quick Test: PubMedQA with CONTEXTS corpus"
echo "============================================================"

# Step 1: 准备corpus目录
echo "Step 1: Preparing corpus..."
mkdir -p dataset/pubmed_pubmedqa/chunk
cp pubmedqa_contexts_chunks.jsonl dataset/pubmed_pubmedqa/chunk/pubmed.jsonl

# Step 2: 构建图（500 passages）
echo ""
echo "Step 2: Building graph (this will take 5-10 minutes)..."
python run.py \\
    --dataset_name pubmed_pubmedqa \\
    --dataset pubmedqa \\
    --mirage_dataset pubmedqa \\
    --llm_name gpt-3.5-turbo \\
    --retrieval_method linearrag \\
    --top_k 32 \\
    --build_graph

# Step 3: 运行测试（前50个问题）
echo ""
echo "Step 3: Running test on first 50 questions..."
python run.py \\
    --dataset_name pubmed_pubmedqa \\
    --dataset pubmedqa \\
    --mirage_dataset pubmedqa \\
    --llm_name gpt-3.5-turbo \\
    --retrieval_method linearrag \\
    --top_k 32 \\
    --max_samples 50

echo ""
echo "============================================================"
echo "Test complete! Check the results."
echo "Expected improvements:"
echo "  - Retrieval scores: 0.001 -> 0.1-0.3 (100x-300x better)"
echo "  - Accuracy: ~0% -> 60-80%"
echo "============================================================"
"""
    
    with open(output_file, 'w') as f:
        f.write(script)
    
    os.chmod(output_file, 0o755)
    
    print(f"\n✅ Created quick test script: {output_file}")
    print(f"   Run with: ./{output_file}")

def main():
    parser = argparse.ArgumentParser(description='Add PubMedQA chunks to existing graph')
    parser.add_argument('--graph-dir', type=str, default='import/pubmed_mirage_medqa',
                        help='Existing graph directory')
    parser.add_argument('--chunks-file', type=str, default='pubmedqa_contexts_chunks.jsonl',
                        help='PubMedQA chunks file')
    
    args = parser.parse_args()
    
    print("\n🚀 PubMedQA Graph Augmentation Pipeline\n")
    
    # Step 1: 检查现有图
    graph_info = load_existing_graph_info(args.graph_dir)
    
    # Step 2: 加载PubMedQA chunks
    pubmedqa_chunks = load_pubmedqa_chunks(args.chunks_file)
    
    if not pubmedqa_chunks:
        print("\n❌ Failed to load PubMedQA chunks")
        return
    
    # Step 3: 创建快速测试脚本
    create_quick_test_script()
    
    # 总结
    print("\n" + "="*80)
    print("Summary & Recommendations")
    print("="*80)
    
    print("""
我们有两个方案可以测试：

方案A: 快速验证 - 只用500个PubMedQA chunks (推荐先做)
====================================================
优点：
  ✅ 快速（5-10分钟构建图）
  ✅ 能验证核心假设：LinearRAG能否从正确的corpus检索到对应文档
  ✅ 100% corpus覆盖（所有500个问题的CONTEXTS都在corpus中）

步骤：
  1. 运行: ./test_pubmedqa_with_contexts.sh
  2. 查看检索分数是否提升（0.001 -> 0.1+）
  3. 查看准确率是否提升（0% -> 60-80%）

预期结果：
  如果假设正确，应该看到：
  - 检索分数显著提升（100-300倍）
  - 准确率从0%提升到60-80%
  - 证明corpus匹配的重要性


方案B: 完整测试 - 50k + 500 chunks
===================================
优点：
  ✅ 更真实的场景（大规模corpus）
  ✅ 测试LinearRAG在大corpus中的检索能力

缺点：
  ❌ 需要重建50k的图（几小时）
  ❌ 计算成本高

步骤：
  1. 将pubmedqa_contexts_chunks.jsonl追加到原始corpus
  2. 删除旧图: rm -rf import/pubmed_mirage_medqa
  3. 重建图: python run.py --dataset_name pubmed --dataset medqa --build_graph
  4. 测试: python run.py --dataset pubmedqa --mirage_dataset pubmedqa


建议顺序：
=========
1. 先运行方案A（快速验证，10分钟）
2. 如果方案A成功，说明假设正确
3. 再考虑是否运行方案B（完整测试，几小时）

方案A足以回答核心问题：
  "如果corpus包含正确文档，LinearRAG能否检索到？"
""")
    
    print("\n" + "="*80)
    print("Ready to test! Run: ./test_pubmedqa_with_contexts.sh")
    print("="*80)

if __name__ == '__main__':
    main()
