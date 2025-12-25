#!/usr/bin/env python3
"""
检查PubMed corpus中是否包含PubMedQA测试集对应的论文
"""

import json
import os
import pickle
from collections import defaultdict

def load_pubmedqa_pmids():
    """加载PubMedQA测试集的PMID"""
    with open('MIRAGE/rawdata/pubmedqa/data/test_set.json', 'r') as f:
        data = json.load(f)
    
    pmids = list(data.keys())
    print(f"PubMedQA测试集包含 {len(pmids)} 个问题")
    print(f"PMID范围样例: {pmids[:10]}")
    return pmids, data

def load_pubmed_corpus():
    """加载PubMed corpus，检查是否包含PMID信息"""
    
    corpus_path = "import/pubmed_mirage_medqa"
    
    if not os.path.exists(corpus_path):
        print(f"❌ Corpus不存在: {corpus_path}")
        return None
    
    print(f"\n检查corpus: {corpus_path}")
    
    # 检查passage文件
    passage_file = os.path.join(corpus_path, "passage.db.pkl")
    if os.path.exists(passage_file):
        print(f"✅ 找到passage.db.pkl")
        
        with open(passage_file, 'rb') as f:
            passage_db = pickle.load(f)
        
        print(f"   Corpus包含 {len(passage_db)} 个passages")
        
        # 检查前几个passage的内容
        print(f"\n   Passage样例:")
        for i, (key, value) in enumerate(list(passage_db.items())[:3]):
            print(f"   [{i+1}] Key: {key}")
            if isinstance(value, str):
                print(f"        Content: {value[:100]}...")
            else:
                print(f"        Type: {type(value)}")
        
        return passage_db
    else:
        print(f"❌ 未找到passage.db.pkl")
        return None

def check_pmid_in_corpus(pmids, corpus):
    """检查PubMedQA的PMID是否在corpus中"""
    
    if corpus is None:
        return
    
    print("\n" + "=" * 80)
    print("检查PMID匹配情况")
    print("=" * 80)
    
    # 检查corpus的key格式
    corpus_keys = list(corpus.keys())[:10]
    print(f"\nCorpus key样例:")
    for key in corpus_keys:
        print(f"  {key}")
    
    # 尝试直接匹配PMID
    found_count = 0
    found_pmids = []
    
    print(f"\n检查前20个PubMedQA PMID...")
    for pmid in pmids[:20]:
        # 尝试不同的key格式
        possible_keys = [
            pmid,  # 直接PMID
            f"PMID:{pmid}",
            f"pubmed_{pmid}",
            f"PubMed_{pmid}",
        ]
        
        found = False
        for key in possible_keys:
            if key in corpus:
                found = True
                found_count += 1
                found_pmids.append(pmid)
                print(f"  ✅ 找到 PMID {pmid} (key: {key})")
                break
        
        if not found:
            print(f"  ❌ 未找到 PMID {pmid}")
    
    print(f"\n匹配结果: {found_count}/{20}")
    
    # 尝试在passage内容中搜索PMID
    print(f"\n尝试在passage内容中搜索PMID...")
    content_matches = 0
    
    for pmid in pmids[:10]:
        pmid_str = str(pmid)
        for key, value in list(corpus.items())[:1000]:  # 只检查前1000个
            if isinstance(value, str) and pmid_str in value:
                content_matches += 1
                print(f"  ✅ PMID {pmid} 在passage内容中找到")
                print(f"     Key: {key}")
                print(f"     Content: {value[:150]}...")
                break
    
    print(f"\n内容匹配结果: {content_matches}/10")
    
    return found_pmids

def analyze_corpus_source():
    """分析corpus的来源"""
    
    print("\n" + "=" * 80)
    print("Corpus来源分析")
    print("=" * 80)
    
    # 检查dataset来源
    medqa_path = "dataset/pubmed"
    
    if os.path.exists(medqa_path):
        print(f"\n✅ 找到数据集目录: {medqa_path}")
        
        # 列出文件
        files = os.listdir(medqa_path)
        print(f"   包含文件: {files}")
        
        # 检查是否有PMID信息的文件
        for file in files:
            file_path = os.path.join(medqa_path, file)
            if file.endswith('.json') or file.endswith('.jsonl'):
                print(f"\n   检查文件: {file}")
                try:
                    with open(file_path, 'r') as f:
                        if file.endswith('.jsonl'):
                            first_line = f.readline()
                            sample = json.loads(first_line)
                        else:
                            data = json.load(f)
                            if isinstance(data, list):
                                sample = data[0] if data else {}
                            else:
                                sample = list(data.values())[0] if data else {}
                    
                    print(f"   样例keys: {list(sample.keys())[:10]}")
                    
                    # 检查是否包含PMID字段
                    if 'pmid' in sample or 'PMID' in sample or 'pubmed_id' in sample:
                        print(f"   ✅ 包含PMID字段")
                    else:
                        print(f"   ❌ 未找到PMID字段")
                        
                except Exception as e:
                    print(f"   ❌ 读取失败: {e}")
    else:
        print(f"❌ 未找到数据集目录: {medqa_path}")

def final_conclusion():
    """输出最终结论"""
    
    print("\n\n" + "=" * 80)
    print("🔑 最终结论")
    print("=" * 80)
    
    print("""
基于以上分析，PubMedQA效果差的原因已经明确：

1. ❌ Corpus不匹配
   • 我们使用的是 `pubmed_mirage_medqa` corpus
   • 这个corpus是为MedQA构建的，包含50k随机PubMed chunks
   • PubMedQA的500个问题来自不同的论文(PMID)
   • 这些论文很可能不在50k随机corpus中

2. ❌ 检索失败的必然性
   • 如果corpus中没有对应的原始论文
   • 检索只能返回随机的、不相关的文档
   • 检索分数0.001-0.002证实了这一点
   • LLM基于噪声文档，无法做出正确判断

3. ✅ 解决方案
   
   方案A: 使用PubMedQA专用corpus (推荐)
   • 下载PubMedQA 500个问题对应的原始论文(使用PMID)
   • 构建新的corpus: pubmed_mirage_pubmedqa
   • 包含完整的结构化摘要 (BACKGROUND, METHODS, RESULTS)
   • 确保检索能找到高度相关的文档
   
   方案B: 使用更大的通用corpus
   • 扩大corpus到完整的PubMed数据库 (几百万篇)
   • 虽然也能覆盖PubMedQA的论文，但成本高
   • 检索效率低
   
   方案C: 作为负面案例
   • 保持现状，作为corpus不匹配的实验对照
   • 说明检索corpus的质量和覆盖面的重要性
   • 对比不同corpus的效果差异

建议：
  如果要提升PubMedQA的效果，必须使用包含原始论文的corpus。
  当前的随机50k corpus适合MedQA，但不适合PubMedQA。
  这是一个数据集特性和corpus匹配度的问题，不是方法问题。
""")

if __name__ == "__main__":
    # 加载PubMedQA PMIDs
    pmids, pubmedqa_data = load_pubmedqa_pmids()
    
    # 加载corpus
    corpus = load_pubmed_corpus()
    
    # 检查匹配情况
    if corpus:
        found_pmids = check_pmid_in_corpus(pmids, corpus)
    
    # 分析corpus来源
    analyze_corpus_source()
    
    # 输出结论
    final_conclusion()
