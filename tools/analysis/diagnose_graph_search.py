#!/usr/bin/env python3
"""
诊断LinearRAG图搜索问题
"""
import sys
sys.path.insert(0, '/home/maoxy23/projects/LinearRAG')

from src.config import LinearRAGConfig
from src.LinearRAG import LinearRAG
from sentence_transformers import SentenceTransformer
import json

# 加载配置
config = LinearRAGConfig(
    embedding_model=SentenceTransformer('model/all-mpnet-base-v2'),
    dataset_name='pubmed_pubmedqa',
    working_dir='import',
    retrieval_top_k=3
)

# 加载LinearRAG（跳过index，直接加载现有的）
print("加载LinearRAG...")
rag = LinearRAG(global_config=config)

# 加载现有的图和embeddings
print("加载现有的图和embeddings...")
rag.load_index()

print(f"\n图统计:")
print(f"  节点数: {rag.graph.vcount()}")
print(f"  边数: {rag.graph.ecount()}")
print(f"  Passage节点数: {len(rag.passage_node_indices)}")
print(f"  Entity数: {len(rag.entity_hash_ids)}")

# 测试第1个问题
with open('MIRAGE/rawdata/pubmedqa.json', 'r') as f:
    pubmedqa = json.load(f)

first_pmid = list(pubmedqa.keys())[0]
first_q = pubmedqa[first_pmid]
question = first_q['QUESTION']

print(f"\n测试问题:")
print(f"  PMID: {first_pmid}")
print(f"  问题: {question}")

# 检索
print(f"\n执行检索...")
result = rag.retrieve([{"question": question, "answer": first_q['final_decision']}])

print(f"\n检索结果:")
print(f"  has_entities: {result[0]['has_entities']}")
print(f"  检索分数: {result[0]['sorted_passage_scores']}")

# 显示检索到的文档
print(f"\n检索到的前3个文档:")
for i, (doc, score) in enumerate(zip(result[0]['sorted_passage'], result[0]['sorted_passage_scores'])):
    print(f"\n文档{i+1} (分数: {score:.6f}):")
    print(f"  {doc[:200]}...")

# 检查正确文档是否在corpus中
with open('dataset/pubmed_pubmedqa/chunk/pubmed.jsonl', 'r') as f:
    corpus = [json.loads(line) for line in f]

correct_doc = None
for chunk in corpus:
    if chunk.get('pmid') == first_pmid:
        correct_doc = chunk
        break

if correct_doc:
    print(f"\n正确文档:")
    print(f"  PMID: {correct_doc['pmid']}")
    print(f"  内容开头: {correct_doc['text'][:200]}...")
    
    # 检查是否被检索到
    found = False
    for doc in result[0]['sorted_passage']:
        if correct_doc['text'][:100] in doc or doc[:100] in correct_doc['text']:
            found = True
            break
    
    print(f"  ✅ 被检索到" if found else "  ❌ 未被检索到")
else:
    print(f"\n❌ 正确文档不在corpus中")

# 检查passage_hash_id映射
print(f"\n检查hash_id映射:")
print(f"  passage_hash_id_to_text有 {len(rag.passage_embedding_store.hash_id_to_text)} 个映射")
sample_hash_ids = list(rag.passage_embedding_store.hash_id_to_text.keys())[:3]
print(f"  前3个hash_id: {[h[:40]+'...' for h in sample_hash_ids]}")
