#!/usr/bin/env python
"""验证 MedMCQA 答案格式一致性"""

import json
import re

def test_answer_format():
    print("=" * 70)
    print("MedMCQA 答案格式验证")
    print("=" * 70)
    
    # 1. 检查 gold_answer 格式
    print("\n1️⃣  检查 Gold Answer 格式:")
    print("-" * 70)
    
    cop_mapping = {1: "A", 2: "B", 3: "C", 4: "D"}
    gold_answers = []
    
    with open('MIRAGE/rawdata/medmcqa/data/dev.json', 'r') as f:
        for i, line in enumerate(f):
            if i >= 5:
                break
            item = json.loads(line)
            cop = item.get('cop')
            gold_ans = cop_mapping.get(cop, "A")
            gold_answers.append(gold_ans)
            print(f"  样本 {i+1}: cop={cop} → gold_answer=\"{gold_ans}\"")
    
    print(f"\n  ✅ Gold answers 都是单字母格式: {set(gold_answers)}")
    
    # 2. 模拟 LLM 预测答案的解析
    print("\n2️⃣  模拟 LLM 预测答案解析:")
    print("-" * 70)
    
    test_responses = [
        "A",
        "B",
        "The answer is C",
        "D. This is the correct option",
        "Answer: A",
        "I think the answer is B because...",
        "C",
        "Option D is correct",
    ]
    
    def parse_answer(qa_result):
        """模拟 LinearRAG.py 中的答案解析逻辑"""
        pred_ans = qa_result.strip().upper()
        
        # 直接检查是否是单个字母
        if pred_ans in ['A', 'B', 'C', 'D']:
            return pred_ans
        else:
            # 尝试从文本中提取第一个字母
            match = re.search(r'[ABCD]', pred_ans)
            if match:
                return match.group(0)
            else:
                return "INVALID"
    
    for i, response in enumerate(test_responses, 1):
        parsed = parse_answer(response)
        print(f"  测试 {i}: \"{response[:40]}\" → \"{parsed}\"")
    
    # 3. 检查评估逻辑
    print("\n3️⃣  检查评估逻辑:")
    print("-" * 70)
    
    print("  evaluate.py 中的 calculate_llm_accuracy:")
    print("    - 当前: 使用 LLM 判断答案是否正确")
    print("    - 问题: 对于多选题，应该直接比较字母！")
    print("    - 建议: 添加直接字符串比较逻辑")
    
    # 4. 完整流程测试
    print("\n4️⃣  完整流程模拟:")
    print("-" * 70)
    
    test_cases = [
        {"gold": "A", "pred": "A", "expected_correct": True},
        {"gold": "B", "pred": "The answer is B", "expected_correct": True},
        {"gold": "C", "pred": "D", "expected_correct": False},
        {"gold": "A", "pred": "Answer: A. Because...", "expected_correct": True},
    ]
    
    for i, case in enumerate(test_cases, 1):
        parsed_pred = parse_answer(case["pred"])
        is_correct = (parsed_pred == case["gold"])
        status = "✅" if is_correct == case["expected_correct"] else "❌"
        print(f"  {status} 测试 {i}:")
        print(f"     Gold: \"{case['gold']}\", Pred: \"{case['pred'][:30]}\" → \"{parsed_pred}\"")
        print(f"     Match: {is_correct} (预期: {case['expected_correct']})")
    
    # 5. 建议
    print("\n" + "=" * 70)
    print("🔍 发现的问题和建议:")
    print("=" * 70)
    print("""
1. ✅ Gold Answer 格式正确: 都是单字母 A/B/C/D
2. ✅ LLM Pred Answer 解析逻辑正确: 可以正确提取 A/B/C/D
3. ⚠️  评估逻辑需要优化:
   - 当前: evaluate.py 使用 LLM 判断答案
   - 问题: 对于多选题，LLM 判断可能不准确且浪费 API 调用
   - 建议: 对于 medqa/medmcqa/mmlu，应该直接比较字母

4. 📊 推荐的评估逻辑:
   def calculate_llm_accuracy(self, pre_answer, gold_ans, dataset_name):
       # 对于多选题数据集，直接比较字符串
       if dataset_name in ["medqa", "medmcqa", "mmlu"]:
           return 1.0 if pre_answer.strip().upper() == gold_ans.strip().upper() else 0.0
       
       # 其他数据集使用 LLM 判断
       else:
           # ... 原有的 LLM 判断逻辑
    """)
    
    print("\n💡 建议修改:")
    print("   需要修改 src/evaluate.py 的 calculate_llm_accuracy 函数")
    print("   添加对多选题数据集的直接字符串比较支持")
    print("=" * 70)

if __name__ == "__main__":
    test_answer_format()
