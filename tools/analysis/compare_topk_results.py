#!/usr/bin/env python3
"""
对比分析：Top-K=5 vs Top-K=3 的效果
"""
import json
from collections import Counter

def analyze_results(filepath, label):
    """分析单个结果文件"""
    print(f"\n{'='*80}")
    print(f"{label}")
    print(f"{'='*80}")
    
    try:
        with open(filepath, 'r') as f:
            results = json.load(f)
    except FileNotFoundError:
        print(f"❌ 文件不存在: {filepath}")
        return None
    
    print(f"\n共 {len(results)} 个问题")
    
    # 1. 预测答案分布
    pred_answers = [r['pred_answer'] for r in results]
    answer_dist = Counter(pred_answers)
    
    print(f"\n📊 预测答案分布:")
    for answer, count in answer_dist.most_common():
        print(f"   {answer}: {count} ({count/len(results)*100:.1f}%)")
    
    # 2. 检索分数统计
    all_scores = []
    for r in results:
        if 'sorted_passage_scores' in r:
            all_scores.extend(r['sorted_passage_scores'])
    
    if all_scores:
        print(f"\n🎯 检索分数统计:")
        print(f"   平均分: {sum(all_scores)/len(all_scores):.6f}")
        print(f"   最高分: {max(all_scores):.6f}")
        print(f"   最低分: {min(all_scores):.6f}")
        print(f"   中位数: {sorted(all_scores)[len(all_scores)//2]:.6f}")
    
    # 3. 检查第1个问题的检索质量
    if results:
        first = results[0]
        print(f"\n🔍 第1个问题检索示例:")
        print(f"   问题: {first['question'][:60]}...")
        print(f"   正确答案: {first['answer']}")
        print(f"   预测答案: {first['pred_answer']}")
        if 'sorted_passage_scores' in first:
            print(f"   检索分数: {first['sorted_passage_scores']}")
        
        # 检查检索到的文档是否相关
        if 'sorted_passage' in first and first['sorted_passage']:
            first_doc = first['sorted_passage'][0]
            question_lower = first['question'].lower()
            doc_lower = first_doc[:200].lower()
            
            # 提取问题中的关键医学术语
            keywords = []
            for word in question_lower.split():
                if len(word) > 5 and word.isalpha():
                    keywords.append(word)
            
            relevance_score = sum(1 for kw in keywords if kw in doc_lower)
            print(f"   相关度评估: {relevance_score}/{len(keywords)} 关键词匹配")
            print(f"   第1个文档: {first_doc[:150]}...")
    
    return {
        'total': len(results),
        'pred_dist': answer_dist,
        'avg_score': sum(all_scores)/len(all_scores) if all_scores else 0,
        'max_score': max(all_scores) if all_scores else 0,
    }

def main():
    print("="*80)
    print("PubMedQA Results Comparison: Top-K=5 vs Top-K=3")
    print("="*80)
    
    # 分析之前的结果 (top-k=5, 500个问题)
    old_stats = analyze_results(
        'results_pubmed_pubmedqa_pubmedqa.json',
        '📋 之前的测试 (Top-K=5, 500个问题)'
    )
    
    # 分析新的结果 (top-k=3, 50个问题)
    new_stats = analyze_results(
        'results_pubmed_pubmedqa_pubmedqa.json',
        '📋 修复后的测试 (Top-K=3, 50个问题)'
    )
    
    # 对比分析
    if old_stats and new_stats:
        print(f"\n{'='*80}")
        print("📊 对比分析")
        print(f"{'='*80}")
        
        print(f"\n检索质量对比:")
        print(f"   之前平均分: {old_stats['avg_score']:.6f}")
        print(f"   修复后平均分: {new_stats['avg_score']:.6f}")
        print(f"   变化: {(new_stats['avg_score']/old_stats['avg_score']-1)*100:+.1f}%")
        
        print(f"\n预测多样性对比:")
        print(f"   之前: {len(old_stats['pred_dist'])} 种答案")
        print(f"   修复后: {len(new_stats['pred_dist'])} 种答案")
        
        # 期望改进
        print(f"\n期望改进:")
        print(f"   ✅ Top-K从5→3应该减少噪声")
        print(f"   ✅ 检索分数应该有显著提升（如果corpus正确）")
        print(f"   ✅ 预测答案应该更多样化（不全是Maybe）")

if __name__ == '__main__':
    main()
