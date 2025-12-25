"""
分析NER未能提取实体的问题
"""

# 从日志中提取的未提取实体的问题样本
failed_questions = [
    "Which cells in the blood do not have a nucleus?",
    "The cardiac cycle consists of the phases:",
    "The chain of survival has four links. Put the following list in the correct order...",
    "What three factors regulate stroke volume?",
    "Name the bones of the middle finger in the correct order from the hand.",
    "Which of the following would not be done before catheterizing?",
    "When developing a plan of care relating to the management of a person's pain, at...",
    "Who is the publication Your guide to the NHS written for?",
    "Name three of the five main uses of the hand.",
    "The process of translation requires the presence of:",
    "The enzymes of glycolysis are located in the:",
    "The most rapid method to resynthesize ATP during exercise is through:",
    "Eccrine and apocrine glands are both types of:",
    "The energy for all forms of muscle contraction is provided by:",
    "The coding sequences of genes are called:",
    "Which of the following is an example of monosomy?",
]

print("分析问题特点：\n")
print("1. 这些问题的共同特点：")
print("   - 大多是简短的定义性问题")
print("   - 包含常见的生物/医学术语（如 ATP, glycolysis, monosomy）")
print("   - 很多是功能性描述（如 'cells that do not have nucleus'）")
print("   - 结构简单，缺少上下文\n")

print("2. 可能的NER失败原因：")
print("   - 生物医学NER模型可能对这些常见术语的识别不够敏感")
print("   - 短文本缺少足够的上下文信息")
print("   - 问题本身就是在询问实体名称，而不是包含实体")
print("   - 某些通用词汇（如 'hand', 'blood', 'cells'）可能被过滤\n")

print("3. 建议的解决方案：")
