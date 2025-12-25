#!/usr/bin/env python3
"""
方案2: 直接使用PubMedQA自带的CONTEXTS作为corpus
这是最快速的方法，因为CONTEXTS已经是高质量的结构化摘要
"""

import json
import os
from collections import defaultdict

def extract_contexts_as_chunks(data_file='MIRAGE/rawdata/pubmedqa/data/test_set.json'):
    """
    从PubMedQA数据集中提取CONTEXTS作为chunks
    每个PMID的CONTEXTS组合成一个chunk
    """
    
    print("="*80)
    print("Extracting CONTEXTS from PubMedQA")
    print("="*80)
    
    with open(data_file, 'r') as f:
        data = json.load(f)
    
    chunks = []
    
    for pmid, item in data.items():
        question = item['QUESTION']
        contexts = item['CONTEXTS']
        labels = item.get('LABELS', [])
        answer = item['final_decision']
        long_answer = item.get('LONG_ANSWER', '')
        
        # 方案A: 将所有CONTEXTS合并为一个chunk
        combined_text = ' '.join(contexts)
        
        chunk = {
            'pmid': pmid,
            'text': combined_text,
            'question': question,
            'answer': answer,
            'long_answer': long_answer,
            'contexts': contexts,  # 保留原始结构
            'labels': labels,
            'source': 'pubmedqa_contexts',
        }
        
        chunks.append(chunk)
    
    print(f"\n✅ Extracted {len(chunks)} chunks from PubMedQA CONTEXTS")
    
    # 统计
    total_chars = sum(len(c['text']) for c in chunks)
    avg_chars = total_chars / len(chunks)
    
    print(f"   Average chunk length: {avg_chars:.0f} characters")
    print(f"   Total text: {total_chars:,} characters")
    
    return chunks

def save_chunks_jsonl(chunks, output_file='pubmedqa_contexts_chunks.jsonl'):
    """保存为JSONL格式"""
    
    with open(output_file, 'w') as f:
        for chunk in chunks:
            f.write(json.dumps(chunk) + '\n')
    
    print(f"✅ Saved to: {output_file}")
    
    return output_file

def merge_with_original_corpus(pubmedqa_chunks_file, 
                                original_corpus_file='dataset/pubmed/chunk/pubmed.jsonl',
                                output_file='dataset/pubmed/chunk/pubmed_with_pubmedqa.jsonl'):
    """
    将PubMedQA chunks与原始50k corpus合并
    """
    
    print("\n" + "="*80)
    print("Merging with Original Corpus")
    print("="*80)
    
    # 读取原始corpus
    if not os.path.exists(original_corpus_file):
        print(f"⚠️  Warning: Original corpus not found at {original_corpus_file}")
        print(f"   Will create new corpus with only PubMedQA chunks")
        original_chunks = []
    else:
        original_chunks = []
        with open(original_corpus_file, 'r') as f:
            for line in f:
                original_chunks.append(json.loads(line))
        print(f"✅ Loaded original corpus: {len(original_chunks)} chunks")
    
    # 读取PubMedQA chunks
    pubmedqa_chunks = []
    with open(pubmedqa_chunks_file, 'r') as f:
        for line in f:
            pubmedqa_chunks.append(json.loads(line))
    print(f"✅ Loaded PubMedQA chunks: {len(pubmedqa_chunks)} chunks")
    
    # 合并
    all_chunks = original_chunks + pubmedqa_chunks
    print(f"\n📊 Total chunks after merge: {len(all_chunks)}")
    print(f"   Original: {len(original_chunks)}")
    print(f"   PubMedQA: {len(pubmedqa_chunks)}")
    
    # 保存
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w') as f:
        for chunk in all_chunks:
            f.write(json.dumps(chunk) + '\n')
    
    print(f"\n✅ Saved merged corpus to: {output_file}")
    
    return output_file

def show_sample_chunks(chunks_file, n=3):
    """显示样例chunks"""
    
    print("\n" + "="*80)
    print("Sample Chunks")
    print("="*80)
    
    with open(chunks_file, 'r') as f:
        chunks = [json.loads(line) for line in f]
    
    for i, chunk in enumerate(chunks[:n]):
        print(f"\nChunk {i+1}:")
        print(f"  PMID: {chunk.get('pmid', 'N/A')}")
        print(f"  Question: {chunk.get('question', 'N/A')[:80]}...")
        print(f"  Answer: {chunk.get('answer', 'N/A')}")
        
        if 'labels' in chunk:
            print(f"  Structure: {' -> '.join(chunk['labels'])}")
        
        print(f"  Text length: {len(chunk['text'])} chars")
        print(f"  Text preview: {chunk['text'][:200]}...")
        print("-"*80)

def main():
    """
    主流程：
    1. 从PubMedQA提取CONTEXTS作为chunks
    2. 保存为JSONL格式
    3. 与原始50k corpus合并
    4. 显示样例
    """
    
    print("\n🚀 PubMedQA CONTEXTS Extraction Pipeline\n")
    
    # Step 1: 提取CONTEXTS
    chunks = extract_contexts_as_chunks()
    
    # Step 2: 保存chunks
    chunks_file = save_chunks_jsonl(chunks)
    
    # Step 3: 显示样例
    show_sample_chunks(chunks_file)
    
    # Step 4: 与原始corpus合并
    merged_file = merge_with_original_corpus(chunks_file)
    
    # 总结
    print("\n" + "="*80)
    print("✅ Pipeline Complete!")
    print("="*80)
    
    print(f"""
Files created:
  1. PubMedQA chunks: {chunks_file}
  2. Merged corpus: {merged_file}

Next steps to rebuild graph with augmented corpus:

方案A: 使用现有的pubmed_mirage_medqa图，添加500个chunks
-------------------------------------------------------
1. 将merged corpus放到正确位置:
   cp {merged_file} dataset/pubmed/chunk/pubmed.jsonl
   
2. 删除旧图（触发重建）:
   rm -rf import/pubmed_mirage_medqa
   
3. 重新运行MedQA（会自动重建图）:
   python run.py --dataset_name pubmed \\
                 --dataset medqa \\
                 --mirage_dataset medqa \\
                 --build_graph

4. 然后测试PubMedQA（复用这个图）:
   python run.py --dataset_name pubmed \\
                 --dataset pubmedqa \\
                 --mirage_dataset pubmedqa \\
                 --llm_name gpt-3.5-turbo \\
                 --retrieval_method linearrag \\
                 --top_k 32

方案B: 先用500个PubMedQA chunks测试（快速验证）
------------------------------------------------
1. 创建小型测试corpus:
   head -n 500 {chunks_file} > dataset/pubmed/chunk/pubmed_test.jsonl
   
2. 修改run.py指向测试corpus

3. 构建小图测试

这样可以快速验证：
  - 如果corpus包含正确文档，检索分数是否提升？
  - LinearRAG能否筛选出对应的CONTEXTS？
  - 准确率是否显著提升？

预期结果：
  - 检索分数: 0.001 → 0.1-0.3 (提升100-300倍)
  - 准确率: ~0% → 60-80%
""")

if __name__ == '__main__':
    main()
