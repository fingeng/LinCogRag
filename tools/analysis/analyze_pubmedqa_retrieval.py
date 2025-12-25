#!/usr/bin/env python3
"""
分析PubMedQA检索质量
对比真实CONTEXTS和检索到的文档
"""

import json
import os
from collections import Counter

def analyze_contexts_characteristics():
    """分析PubMedQA数据集中CONTEXTS的特征"""
    
    print("=" * 80)
    print("PubMedQA CONTEXTS 特征分析")
    print("=" * 80)
    
    with open('MIRAGE/rawdata/pubmedqa/data/test_set.json', 'r') as f:
        data = json.load(f)
    
    # 统计CONTEXTS特征
    contexts_per_question = []
    total_contexts_length = []
    label_distribution = Counter()
    
    for pmid, item in data.items():
        contexts = item['CONTEXTS']
        labels = item.get('LABELS', [])
        
        contexts_per_question.append(len(contexts))
        for ctx in contexts:
            total_contexts_length.append(len(ctx))
        
        label_distribution.update(labels)
    
    print(f"\n数据集规模: {len(data)} 个问题")
    print(f"\nCONTEXTS 数量分布:")
    print(f"  平均每题: {sum(contexts_per_question)/len(contexts_per_question):.2f} 段")
    print(f"  范围: {min(contexts_per_question)} - {max(contexts_per_question)} 段")
    
    print(f"\nCONTEXTS 长度分布:")
    print(f"  平均长度: {sum(total_contexts_length)/len(total_contexts_length):.0f} 字符")
    print(f"  范围: {min(total_contexts_length)} - {max(total_contexts_length)} 字符")
    
    print(f"\nCONTEXTS 类型分布 (Top 10):")
    for label, count in label_distribution.most_common(10):
        print(f"  {label}: {count} ({count/sum(label_distribution.values())*100:.1f}%)")
    
    # 分析CONTEXTS内容特征
    print(f"\n\nCONTEXTS 内容特征:")
    print("=" * 80)
    
    # 随机选择5个样本进行详细分析
    import random
    sample_pmids = random.sample(list(data.keys()), 5)
    
    for i, pmid in enumerate(sample_pmids, 1):
        item = data[pmid]
        print(f"\n样本 {i} (PMID: {pmid})")
        print(f"问题: {item['QUESTION']}")
        print(f"答案: {item['final_decision']}")
        print(f"CONTEXTS 结构: {' -> '.join(item.get('LABELS', []))}")
        
        print("\n关键发现:")
        contexts = item['CONTEXTS']
        
        # 分析每个context的内容特点
        for j, (ctx, label) in enumerate(zip(contexts, item.get('LABELS', [])), 1):
            print(f"\n  [{label}] 段落 {j}:")
            print(f"    长度: {len(ctx)} 字符")
            
            # 检查是否包含数据、统计结果
            has_numbers = any(char.isdigit() for char in ctx)
            has_stats = any(keyword in ctx.lower() for keyword in ['p<', 'p=', 'p>', 'or=', 'ci', '95%', 'mean', 'median'])
            has_results = any(keyword in ctx.lower() for keyword in ['results', 'found', 'showed', 'demonstrated', 'observed'])
            
            features = []
            if has_numbers:
                features.append("含数值")
            if has_stats:
                features.append("含统计指标")
            if has_results:
                features.append("描述结果")
            
            if features:
                print(f"    特征: {', '.join(features)}")
            
            # 显示前100字符
            print(f"    内容: {ctx[:100]}...")
        
        print("-" * 80)

def analyze_retrieval_quality():
    """分析检索质量问题"""
    
    print("\n\n" + "=" * 80)
    print("检索质量问题分析")
    print("=" * 80)
    
    # 分析我们的检索corpus
    print("\n当前检索Corpus特征:")
    print("-" * 80)
    
    corpus_info = {
        "来源": "PubMed 50k随机chunks",
        "总chunks数": "49,999",
        "总实体数": "212,532",
        "总句子数": "279,428",
        "特点": [
            "随机采样的PubMed摘要和片段",
            "覆盖医学各个领域",
            "非针对性的通用医学文本",
        ]
    }
    
    print(f"Corpus来源: {corpus_info['来源']}")
    print(f"规模: {corpus_info['总chunks数']} chunks, {corpus_info['总实体数']} entities, {corpus_info['总句子数']} sentences")
    print(f"\n特点:")
    for feature in corpus_info['特点']:
        print(f"  • {feature}")
    
    print("\n\nPubMedQA CONTEXTS vs 我们的Corpus:")
    print("-" * 80)
    
    comparison = {
        "PubMedQA CONTEXTS": {
            "来源": "论文的原始结构化摘要",
            "内容": "BACKGROUND -> METHODS -> RESULTS -> CONCLUSION",
            "特点": [
                "✅ 直接来自论文摘要，高度相关",
                "✅ 包含研究设计、方法、数据",
                "✅ 包含具体的统计结果 (p值, OR, CI等)",
                "✅ 结构化，逻辑完整",
                "✅ 针对问题的直接证据",
            ]
        },
        "我们检索的文档": {
            "来源": "50k随机PubMed chunks",
            "内容": "基于实体相似度检索的句子/段落",
            "特点": [
                "❌ 随机chunks，不一定相关",
                "❌ 可能是不同研究的片段",
                "❌ 缺少完整上下文",
                "❌ 缺少研究设计和方法信息",
                "❌ 检索分数低 (0.001-0.002)，噪声多",
            ]
        }
    }
    
    for source, info in comparison.items():
        print(f"\n{source}:")
        print(f"  来源: {info['来源']}")
        print(f"  内容: {info['内容']}")
        print(f"  特点:")
        for feature in info['特点']:
            print(f"    {feature}")

def suggest_improvements():
    """提出改进建议"""
    
    print("\n\n" + "=" * 80)
    print("改进建议")
    print("=" * 80)
    
    suggestions = [
        {
            "问题": "检索Corpus不匹配",
            "原因": "50k随机PubMed chunks不包含PubMedQA问题对应的原始论文",
            "解决方案": [
                "1. 使用PubMedQA提供的原始论文ID(PMID)，从PubMed下载对应论文摘要",
                "2. 构建专门的PubMedQA corpus，包含500个问题对应的论文全文/摘要",
                "3. 或者扩大corpus规模，增加覆盖面",
            ],
            "预期效果": "检索到真正相关的文档，显著提升准确率"
        },
        {
            "问题": "检索方法不适合结构化QA",
            "原因": "实体相似度检索无法捕捉论文的逻辑结构(BACKGROUND->METHODS->RESULTS)",
            "解决方案": [
                "1. 使用完整论文摘要而非chunks",
                "2. 保留摘要的结构化信息 (BACKGROUND, METHODS, RESULTS)",
                "3. 考虑使用更强的语义检索模型(如SciBERT, PubMedBERT)",
            ],
            "预期效果": "检索到包含完整研究信息的文档"
        },
        {
            "问题": "检索分数过低",
            "原因": "实体overlap太少，检索分数0.001-0.002表明几乎无相关性",
            "解决方案": [
                "1. 检查是否正确使用了PubMed corpus",
                "2. 调整检索参数 (top_k, threshold)",
                "3. 使用混合检索 (实体+语义)",
            ],
            "预期效果": "检索到更相关的文档"
        }
    ]
    
    for i, suggestion in enumerate(suggestions, 1):
        print(f"\n问题 {i}: {suggestion['问题']}")
        print(f"原因: {suggestion['原因']}")
        print(f"\n解决方案:")
        for solution in suggestion['解决方案']:
            print(f"  {solution}")
        print(f"\n预期效果: {suggestion['预期效果']}")
        print("-" * 80)
    
    print("\n\n🔑 核心结论:")
    print("=" * 80)
    print("""
PubMedQA效果差的根本原因：

1. ❌ Corpus不匹配
   - PubMedQA的CONTEXTS来自论文原始摘要
   - 我们的corpus是50k随机PubMed chunks
   - 这50k chunks很可能不包含500个测试问题对应的原始论文

2. ❌ 检索质量差
   - 检索分数0.001-0.002，接近随机
   - 说明检索到的都是无关文档
   - LLM基于噪声文档做判断，倾向于输出"maybe"

3. ✅ 要提升效果，必须：
   - 使用包含PubMedQA原始论文的corpus
   - 或者将PubMedQA的500个问题对应的PMID论文加入corpus
   - 保证检索能找到真正相关的论文摘要

建议：
  • 下载PubMedQA 500个问题对应的原始论文 (使用PMID)
  • 将这些论文的结构化摘要加入corpus
  • 或者在更大的PubMed corpus上测试 (如完整的PubMed数据库)
""")

if __name__ == "__main__":
    analyze_contexts_characteristics()
    analyze_retrieval_quality()
    suggest_improvements()
