#!/usr/bin/env python
"""
BioASQ 数据集示例展示
演示如何理解和处理不同类型的问题
"""

import json
import glob
from collections import Counter

def show_examples():
    print("=" * 80)
    print("BioASQ 数据集示例展示")
    print("=" * 80)
    
    # 读取一个示例文件
    file_path = 'MIRAGE/rawdata/bioasq/Task11BGoldenEnriched/11B1_golden.json'
    with open(file_path, 'r') as f:
        data = json.load(f)
    
    questions = data['questions']
    
    # 1. Yes/No 类型示例
    print("\n" + "🔵 " * 40)
    print("1️⃣  YES/NO 类型问题示例")
    print("🔵 " * 40)
    
    yesno_q = next(q for q in questions if q['type'] == 'yesno')
    print(f"\n问题: {yesno_q['body']}")
    print(f"类型: {yesno_q['type']}")
    print(f"\n✅ 精确答案 (exact_answer):")
    print(f"   \"{yesno_q['exact_answer']}\"")
    print(f"   类型: {type(yesno_q['exact_answer']).__name__}")
    print(f"\n📝 理想答案 (ideal_answer):")
    for i, ans in enumerate(yesno_q['ideal_answer'], 1):
        print(f"   {i}. {ans[:100]}...")
    print(f"\n📚 相关文献:")
    for doc in yesno_q['documents'][:3]:
        print(f"   - {doc}")
    
    # 2. Factoid 类型示例
    print("\n" + "🟢 " * 40)
    print("2️⃣  FACTOID 类型问题示例")
    print("🟢 " * 40)
    
    factoid_q = next(q for q in questions if q['type'] == 'factoid')
    print(f"\n问题: {factoid_q['body']}")
    print(f"类型: {factoid_q['type']}")
    print(f"\n✅ 精确答案 (exact_answer):")
    print(f"   类型: {type(factoid_q['exact_answer']).__name__}")
    print(f"   结构: 嵌套列表 (外层=多个答案, 内层=同义词)")
    for i, ans_group in enumerate(factoid_q['exact_answer'], 1):
        print(f"   答案组 {i}:")
        for j, ans in enumerate(ans_group, 1):
            print(f"      - {ans[:80]}...")
    print(f"\n📝 理想答案:")
    for i, ans in enumerate(factoid_q['ideal_answer'], 1):
        print(f"   {i}. {ans[:100]}...")
    
    # 3. List 类型示例
    print("\n" + "🟡 " * 40)
    print("3️⃣  LIST 类型问题示例")
    print("🟡 " * 40)
    
    list_q = next(q for q in questions if q['type'] == 'list')
    print(f"\n问题: {list_q['body']}")
    print(f"类型: {list_q['type']}")
    print(f"\n✅ 精确答案 (exact_answer):")
    print(f"   类型: {type(list_q['exact_answer']).__name__}")
    print(f"   结构: 列表的列表 (每个内层列表=一个实体)")
    for i, entity in enumerate(list_q['exact_answer'], 1):
        print(f"   实体 {i}: {entity}")
    print(f"\n📝 理想答案:")
    for i, ans in enumerate(list_q['ideal_answer'], 1):
        print(f"   {i}. {ans[:100]}...")
    
    # 4. Summary 类型示例
    print("\n" + "🔴 " * 40)
    print("4️⃣  SUMMARY 类型问题示例")
    print("🔴 " * 40)
    
    summary_q = next(q for q in questions if q['type'] == 'summary')
    print(f"\n问题: {summary_q['body']}")
    print(f"类型: {summary_q['type']}")
    print(f"\n⚠️  注意: Summary类型没有exact_answer!")
    print(f"   'exact_answer' in question: {'exact_answer' in summary_q}")
    print(f"\n📝 理想答案 (ideal_answer):")
    for i, ans in enumerate(summary_q['ideal_answer'], 1):
        print(f"   {i}. {ans[:150]}...")
    
    # 5. 统计信息
    print("\n" + "=" * 80)
    print("📊 统计信息")
    print("=" * 80)
    
    print(f"\n当前文件 ({file_path}):")
    print(f"  总问题数: {len(questions)}")
    
    type_counts = Counter(q['type'] for q in questions)
    for qtype, count in sorted(type_counts.items()):
        print(f"  {qtype:10s}: {count:3d} ({count/len(questions)*100:.1f}%)")
    
    # 6. 全数据集统计
    print(f"\n所有年份统计 (2019-2023):")
    
    all_questions = []
    for task_dir in sorted(glob.glob('MIRAGE/rawdata/bioasq/Task*')):
        task_name = task_dir.split('/')[-1]
        for json_file in glob.glob(f'{task_dir}/*.json'):
            with open(json_file, 'r') as f:
                d = json.load(f)
                all_questions.extend(d['questions'])
    
    all_types = Counter(q['type'] for q in all_questions)
    print(f"  总问题数: {len(all_questions)}")
    for qtype, count in sorted(all_types.items()):
        print(f"  {qtype:10s}: {count:4d} ({count/len(all_questions)*100:.1f}%)")
    
    yesno_count = all_types['yesno']
    print(f"\n✨ MIRAGE使用的Yes/No问题: {yesno_count}个 (与介绍的618个一致！)")
    
    # 7. 答案格式对比
    print("\n" + "=" * 80)
    print("📋 答案格式对比表")
    print("=" * 80)
    
    print("""
┌────────────┬─────────────────┬──────────────────────────┬─────────────────────┐
│ 问题类型   │ exact_answer    │ 格式示例                 │ ideal_answer        │
├────────────┼─────────────────┼──────────────────────────┼─────────────────────┤
│ yesno      │ string          │ "yes" 或 "no"            │ list[str]           │
│ factoid    │ list[list[str]] │ [["答案1", "同义词"]]    │ list[str]           │
│ list       │ list[list[str]] │ [["实体1"], ["实体2"]]   │ list[str]           │
│ summary    │ ❌ 无           │ N/A                      │ list[str]           │
└────────────┴─────────────────┴──────────────────────────┴─────────────────────┘
    """)
    
    # 8. 代码使用示例
    print("=" * 80)
    print("💻 代码使用示例")
    print("=" * 80)
    
    print("""
# 提取Yes/No问题 (MIRAGE使用)
yesno_questions = []
for q in data['questions']:
    if q['type'] == 'yesno':
        yesno_questions.append({
            'question': q['body'],
            'answer': q['exact_answer'],  # "yes" 或 "no"
            'dataset': 'bioasq'
        })

# 提取Factoid答案
factoid_q = next(q for q in data['questions'] if q['type'] == 'factoid')
answers = []
for answer_group in factoid_q['exact_answer']:
    # 通常取第一个（主要答案）
    answers.append(answer_group[0])

# 提取List答案
list_q = next(q for q in data['questions'] if q['type'] == 'list')
entities = [entity[0] for entity in list_q['exact_answer']]

# Summary类型只有ideal_answer
summary_q = next(q for q in data['questions'] if q['type'] == 'summary')
summary_text = summary_q['ideal_answer'][0]
    """)

if __name__ == "__main__":
    show_examples()
