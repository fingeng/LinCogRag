# BioASQ 数据集结构详解

## 📚 数据集介绍

**BioASQ** 是一个大规模生物医学语义索引和问答挑战赛，每年举办一次（Task B）。数据集包含从PubMed文献中提取的生物医学问题。

### 目录结构含义

```
MIRAGE/rawdata/bioasq/
├── Task7BGoldenEnriched/   ← 2019年 (第7届)
├── Task8BGoldenEnriched/   ← 2020年 (第8届)
├── Task9BGoldenEnriched/   ← 2021年 (第9届)
├── Task10BGoldenEnriched/  ← 2022年 (第10届)
└── Task11BGoldenEnriched/  ← 2023年 (第11届)
```

**命名规则**:
- `Task[N]BGoldenEnriched`: 第N届BioASQ Task B的黄金标准数据集
- `[N]B[M]_golden.json`: 第N届第M批次（batch）的测试数据

**统计数据**:
- **2019-2023共5年**: 618个Yes/No问题，2310个总问题
- 每年举办4-6个批次（batches）

## 📋 问题类型 (4种)

### 1. **Yes/No 类型** (618个，33%)

**用途**: 判断题，答案只能是"yes"或"no"

**示例**:
```json
{
  "body": "Can losartan reduce brain atrophy in Alzheimer's disease?",
  "type": "yesno",
  "exact_answer": "no",
  "ideal_answer": ["No. 12 months of treatment with losartan was well tolerated..."],
  "documents": ["http://www.ncbi.nlm.nih.gov/pubmed/34687634"],
  "snippets": [...]
}
```

**字段说明**:
- `exact_answer`: **"yes"** 或 **"no"** (字符串)
- `ideal_answer`: 详细解释（列表）

---

### 2. **Factoid 类型** (事实型问题)

**用途**: 要求一个具体的事实性答案（如名称、缩写、定义等）

**示例**:
```json
{
  "body": "What is CHARMS with respect to medical review of predictive modeling?",
  "type": "factoid",
  "exact_answer": [
    ["CHecklist for critical Appraisal and data extraction for systematic Reviews of prediction Modelling Studies (CHARMS)."]
  ],
  "ideal_answer": ["CHARMS stands for CHecklist for critical Appraisal..."]
}
```

**字段说明**:
- `exact_answer`: **嵌套列表** `[[答案1], [答案2], ...]`
  - 外层列表：可能有多个正确答案
  - 内层列表：每个答案的同义词
- `ideal_answer`: 详细解释（列表）

---

### 3. **List 类型** (列表型问题)

**用途**: 要求列出多个答案（如基因列表、药物列表等）

**示例**:
```json
{
  "body": "Which splicing factors have been associated with alternative splicing in PLN R14del hearts?",
  "type": "list",
  "exact_answer": [
    ["Srrm4"],
    ["Nova1"]
  ],
  "ideal_answer": ["Bioinformatical analysis pointed to the tissue-specific splicing factors Srrm4 and Nova1..."]
}
```

**字段说明**:
- `exact_answer`: **列表的列表** `[[实体1], [实体2], ...]`
  - 每个内层列表代表一个实体及其同义词
- `ideal_answer`: 详细解释（列表）

---

### 4. **Summary 类型** (摘要型问题)

**用途**: 要求生成一段摘要性的回答

**示例**:
```json
{
  "body": "Which are the targets of Tirzepatide?",
  "type": "summary",
  "ideal_answer": ["Tirzepatide is a dual incretin hormones glucagon-like peptide 1 (GLP-1) and glucose-dependent insulinotropic polypeptide (GIP) receptor agonist..."]
}
```

**字段说明**:
- **没有 `exact_answer`**，只有 `ideal_answer`
- `ideal_answer`: 完整的摘要答案（列表）

---

## 🗂️ JSON 文件结构

### 顶层结构
```json
{
  "questions": [
    {问题1},
    {问题2},
    ...
  ]
}
```

### 每个问题的通用字段

| 字段 | 类型 | 说明 | 示例 |
|------|------|------|------|
| `id` | string | 问题唯一ID | `"6402c910201352f04a00000c"` |
| `type` | string | 问题类型 | `"yesno"`, `"factoid"`, `"list"`, `"summary"` |
| `body` | string | 问题内容 | `"Can losartan reduce brain atrophy..."` |
| `documents` | list | 相关PubMed文献URL列表 | `["http://www.ncbi.nlm.nih.gov/pubmed/34687634"]` |
| `snippets` | list | 支持答案的文本片段（**MIRAGE已移除**） | `[]` 或包含snippet对象 |
| `ideal_answer` | list | 详细的理想答案 | `["No. 12 months of treatment..."]` |
| `exact_answer` | string/list | 精确答案（类型依问题而定） | Yes/No: `"yes"`; Factoid/List: `[[...]]` |

### Snippets 字段结构（原始BioASQ有，MIRAGE移除）

```json
"snippets": [
  {
    "offsetInBeginSection": 2574,
    "offsetInEndSection": 2784,
    "text": "INTERPRETATION: 12 months of treatment...",
    "beginSection": "abstract",
    "endSection": "abstract",
    "document": "http://www.ncbi.nlm.nih.gov/pubmed/34687634"
  }
]
```

**注意**: MIRAGE基准测试中，`snippets`字段被移除，要求模型从文档中自行检索。

---

## 📊 年度统计

| 年份 | Task | Yes/No | Factoid | List | Summary | 总计 |
|------|------|--------|---------|------|---------|------|
| 2019 | Task7 | 140 | 120 | 120 | 120 | 500 |
| 2020 | Task8 | 152 | 116 | 116 | 116 | 500 |
| 2021 | Task9 | 117 | 127 | 127 | 126 | 497 |
| 2022 | Task10 | 123 | 121 | 121 | 121 | 486 |
| 2023 | Task11 | 86 | 81 | 81 | 79 | 327 |
| **合计** | | **618** | **565** | **565** | **562** | **2310** |

---

## 🎯 MIRAGE 基准测试使用方式

根据介绍，MIRAGE只使用 **Yes/No 问题**（618个）:

### 数据处理要点

1. **过滤问题**: 只保留 `type == "yesno"` 的问题
2. **移除snippets**: MIRAGE移除了 `snippets` 字段，需要从文档中检索
3. **答案格式**: `exact_answer` 为字符串 `"yes"` 或 `"no"`

### 评估指标

对于Yes/No问题，评估：
- **准确率**: 预测的 "yes"/"no" 是否与 `exact_answer` 匹配
- **LLM判断**: 使用LLM比较预测答案和 `ideal_answer` 的一致性

---

## 💡 实际应用示例

### 问题类型识别
```python
import json

with open('11B1_golden.json', 'r') as f:
    data = json.load(f)

# 统计问题类型
from collections import Counter
types = Counter(q['type'] for q in data['questions'])
# 输出: {'yesno': 24, 'factoid': 19, 'list': 11, 'summary': 19}
```

### 提取Yes/No问题
```python
yesno_questions = [
    q for q in data['questions'] 
    if q['type'] == 'yesno'
]

for q in yesno_questions[:3]:
    print(f"Q: {q['body']}")
    print(f"A: {q['exact_answer']}")
    print(f"Explanation: {q['ideal_answer'][0][:100]}...")
    print()
```

### 答案格式理解
```python
# Yes/No: 直接字符串
yesno_ans = "yes"  # 或 "no"

# Factoid/List: 嵌套列表
factoid_ans = [["CHARMS"], ["Checklist"]]  # 多个同义答案
list_ans = [["Srrm4"], ["Nova1"]]  # 多个实体

# Summary: 无exact_answer，只有ideal_answer
summary_ans = None  # 无精确答案
```

---

## 🔧 建议的数据加载代码

```python
def load_bioasq_yesno(task_dirs):
    """加载BioASQ Yes/No问题"""
    questions = []
    
    for task_dir in task_dirs:
        for json_file in glob.glob(f'{task_dir}/*.json'):
            with open(json_file, 'r') as f:
                data = json.load(f)
                
            for q in data['questions']:
                if q['type'] == 'yesno':
                    questions.append({
                        'question': q['body'],
                        'answer': q['exact_answer'],  # "yes" 或 "no"
                        'ideal_answer': q['ideal_answer'][0],
                        'documents': q['documents'],
                        'dataset': 'bioasq'
                    })
    
    return questions
```

---

## 📖 参考资料

- **BioASQ官网**: http://bioasq.org/
- **MIRAGE论文**: 参考MIRAGE基准测试的描述
- **数据年份对应**:
  - Task 7 → 2019年
  - Task 8 → 2020年
  - Task 9 → 2021年
  - Task 10 → 2022年
  - Task 11 → 2023年

---

## ✅ 总结

1. **Task[N]BGoldenEnriched** = 第N届BioASQ挑战赛的测试集
2. **4种问题类型**: yesno, factoid, list, summary
3. **MIRAGE使用**: 只用618个Yes/No问题，移除了snippets
4. **答案格式**:
   - Yes/No: 字符串 `"yes"/"no"`
   - Factoid/List: 嵌套列表 `[[...]]`
   - Summary: 只有 `ideal_answer`
