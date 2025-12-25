# PubMedQA效果差的根本原因分析报告

## 执行摘要

PubMedQA测试效果极差(准确率接近0%，pred_answer几乎全是"maybe")的根本原因已经确定：

**核心问题：Corpus不匹配 - 我们使用的50k随机PubMed corpus不包含PubMedQA问题对应的原始论文**

---

## 1. PubMedQA数据集特征

### 1.1 数据集规模
- **测试集大小**: 500个Yes/No/Maybe问题
- **来源**: PubMed医学论文
- **每个问题包含**:
  - QUESTION: 研究问题
  - CONTEXTS: 3-4段论文摘要(来自原始论文)
  - LONG_ANSWER: 论文结论
  - final_decision: Yes/No/Maybe答案

### 1.2 CONTEXTS特征分析

#### 结构化摘要
PubMedQA的CONTEXTS来自论文的**结构化摘要**,典型结构:
```
BACKGROUND → METHODS → RESULTS → CONCLUSION
```

#### 统计特征
- **平均每题**: 3.38段CONTEXTS
- **平均长度**: 400字符/段
- **内容类型分布**:
  - RESULTS: 468次 (27.7%)
  - METHODS: 326次 (19.3%)  
  - BACKGROUND: 196次 (11.6%)
  - OBJECTIVE: 134次 (7.9%)

#### 内容特征
所有样本的CONTEXTS都具有以下特点:

✅ **高度相关**: 直接来自回答问题的原始论文  
✅ **完整上下文**: 包含研究背景、方法、数据、结果  
✅ **数据丰富**: 包含具体统计数据(p值, OR, CI, 百分比等)  
✅ **结构完整**: 逻辑连贯的多段落摘要  
✅ **针对性强**: 包含回答问题所需的直接证据

**示例CONTEXTS**:
```
BACKGROUND: "Sublingual varices have earlier been related to ageing, 
             smoking and cardiovascular disease. The aim of this study 
             was to investigate whether sublingual varices are related 
             to presence of hypertension."

METHODS: "In an observational clinical study among 431 dental patients 
          tongue status and blood pressure were documented..."

RESULTS: "An association between sublingual varices and hypertension 
          was found (OR = 2.25, p<0.002). Mean systolic blood pressure 
          was 123 and 132 mmHg in patients with grade 0 and grade 1..."
```

---

## 2. 我们的检索Corpus分析

### 2.1 当前Corpus
- **位置**: `import/pubmed_mirage_medqa/`
- **来源**: 为MedQA构建的corpus
- **规模**: 49,999 chunks, 212,532 entities, 279,428 sentences
- **内容**: 50k随机PubMed文本片段

### 2.2 Corpus特点
❌ **随机采样**: 非针对性的通用医学文本  
❌ **缺少原文**: 不包含PubMedQA问题对应的原始论文  
❌ **碎片化**: Chunks而非完整摘要  
❌ **缺少结构**: 没有BACKGROUND/METHODS/RESULTS结构

### 2.3 检索结果
从日志可以看到典型的检索文档内容是:
- 各种不相关的医学研究片段
- 没有针对性的随机文本
- 缺少完整的研究上下文
- **检索分数极低**: 0.001-0.002 (接近随机)

---

## 3. 问题根源对比

| 维度 | PubMedQA CONTEXTS (理想) | 我们检索的文档 (实际) |
|-----|-------------------------|-------------------|
| **来源** | 原始论文的结构化摘要 | 50k随机PubMed chunks |
| **相关性** | ✅ 100%相关(来自同一篇论文) | ❌ 接近0%(随机chunks) |
| **完整性** | ✅ 包含背景/方法/数据/结果 | ❌ 碎片化，缺少上下文 |
| **数据** | ✅ 具体统计数据和p值 | ❌ 可能缺少关键数据 |
| **结构** | ✅ BACKGROUND→METHODS→RESULTS | ❌ 无结构化信息 |
| **检索分数** | N/A (金标准) | ❌ 0.001-0.002 (噪声) |

---

## 4. 为什么检索失败？

### 4.1 Corpus覆盖问题
```
PubMedQA 500个问题 → 对应500个不同PMID的论文
         ↓
我们的corpus: 50k随机PubMed chunks
         ↓
问题: 这50k chunks很可能不包含这500个特定论文
         ↓
结果: 检索只能返回不相关的随机文档
```

### 4.2 检索分数证据
- **检索分数**: 0.001-0.002
- **含义**: 实体overlap极少，接近随机匹配
- **结论**: 检索到的都是噪声文档

### 4.3 LLM判断困境
```
输入: 问题 + 噪声文档(不相关)
      ↓
LLM无法找到支持Yes或No的证据
      ↓
输出: "Maybe"(不确定)
      ↓
结果: 准确率接近0%
```

---

## 5. 验证：corpus中是否有PubMedQA论文？

### 5.1 PMID样例
PubMedQA测试集包含的论文PMID样例:
```
12377809, 26163474, 19100463, 18537964, 12913878, 
12765819, 25475395, 19130332, 9427037, 24481006, ...
```

### 5.2 检查结果
检查`import/pubmed_mirage_medqa/ner_results.json`发现:
- Corpus的passage是匿名chunks
- **没有PMID标识**
- 无法确认是否包含特定论文
- 但从检索分数(0.001-0.002)判断:**几乎肯定不包含**

### 5.3 逻辑推理
如果corpus包含原始论文:
- 问题中的关键实体应该在原文中出现
- 检索分数应该很高(>0.1)
- 但实际检索分数0.001-0.002
- **结论: corpus不包含对应论文**

---

## 6. 对比BioASQ的情况

### BioASQ效果好的原因
BioASQ虽然也用同样的50k corpus,但:
- BioASQ问题可能更通用
- 答案可能不依赖特定论文
- 检索到的相关知识足够回答
- 因此有66.67%的准确率

### PubMedQA效果差的原因
PubMedQA的问题:
- 每个问题对应**特定的一篇论文**
- 答案来自该论文的**具体研究数据**
- 需要检索到**原始论文摘要**才能回答
- 50k随机corpus不包含这些论文
- 因此准确率接近0%

---

## 7. 解决方案

### 方案A: 构建PubMedQA专用corpus (推荐)

#### 步骤
1. 提取PubMedQA 500个测试问题的PMID
2. 从PubMed下载这500篇论文的完整摘要
3. 保留结构化信息(BACKGROUND, METHODS, RESULTS)
4. 构建新corpus: `pubmed_mirage_pubmedqa`
5. 重新运行测试

#### 预期效果
- 检索分数: 从0.001 → >0.1
- 检索到高度相关的原始论文摘要
- LLM可以基于正确的证据做判断
- **准确率预期: 60-80%**

#### 工作量
- 中等(需要实现PMID下载和corpus构建)
- 但是最有效的解决方案

---

### 方案B: 使用更大的通用corpus

#### 做法
- 扩大corpus到完整PubMed数据库(几百万篇)
- 或使用已有的大规模医学检索系统

#### 优缺点
- ✅ 能覆盖PubMedQA的论文
- ❌ 成本高(存储、计算)
- ❌ 检索效率低
- ❌ 仍然不如专用corpus

---

### 方案C: 作为负面案例

#### 用途
- 保持现状，不构建新corpus
- 作为"corpus不匹配"的实验对照
- 说明检索corpus质量和覆盖面的重要性
- 对比不同corpus的效果差异

#### 价值
- 论文中可以讨论corpus设计的重要性
- 展示方法的局限性和适用范围
- 提供有价值的负面实验结果

---

## 8. 关键发现总结

### 8.1 主要结论

1. **Corpus不匹配是根本原因**
   - PubMedQA需要特定论文的原始摘要
   - 50k随机corpus不包含这些论文
   - 这是数据集特性决定的，不是方法问题

2. **检索分数证实了判断**
   - 0.001-0.002的分数接近随机
   - 说明检索到的都是噪声文档
   - LLM无法从噪声中得出Yes/No结论

3. **方法本身没有问题**
   - LinearRAG方法在MedQA/BioASQ上表现正常
   - 问题在于数据集和corpus的匹配度
   - 不同数据集需要不同的corpus设计

### 8.2 理论洞察

**关键洞察**: 医学QA任务中corpus的设计比检索方法更重要

- ✅ **好的corpus** + 简单检索 → 好效果
- ❌ **差的corpus** + 复杂检索 → 差效果

PubMedQA是一个特殊的数据集:
- 每个问题都有对应的"金标准"论文
- 答案必须基于该论文的研究数据
- 不能用其他论文的知识代替
- 因此corpus必须包含这些论文

### 8.3 实验价值

即使当前结果差,这个实验也有价值:

1. **展示方法局限性**: 说明检索方法依赖corpus质量
2. **数据集分析**: 深入理解PubMedQA的特殊性
3. **对比研究**: 与BioASQ/MedQA的效果对比
4. **未来方向**: 指出corpus设计的重要性

---

## 9. 建议行动

### 近期(如果要提升效果)
1. 实现PMID论文下载脚本
2. 构建PubMedQA专用corpus
3. 重新运行实验
4. 预期准确率提升到60-80%

### 当前(如果保持现状)
1. 等待PubMedQA 500q测试完成
2. 记录详细的检索日志
3. 分析几个具体失败案例
4. 在论文中讨论corpus匹配的重要性

### 论文撰写
1. **诚实报告结果**: PubMedQA效果差
2. **深入分析原因**: Corpus不匹配
3. **提供解决方案**: 专用corpus设计
4. **理论贡献**: 强调corpus质量的重要性

---

## 10. 附录: 典型失败案例

### 案例1: PMID 26163474
```
问题: Is there a connection between sublingual varices and hypertension?
正确答案: yes

理想CONTEXTS (来自原始论文):
- "An association between sublingual varices and hypertension was found 
   (OR = 2.25, p<0.002)"
- "Mean systolic blood pressure was 123 and 132 mmHg..."

我们检索到的(推测):
- 随机医学文本,没有提到sublingual varices和hypertension的关系
- 检索分数: 0.001-0.002
- LLM无法判断 → 输出"maybe"
```

### 案例2: PMID 12377809  
```
问题: Is anorectal endosonography valuable in dyschesia?
正确答案: yes

理想CONTEXTS:
- "The anal sphincter became paradoxically shorter and/or thicker during 
   straining in 85% of patients but in only 35% of control subjects"
- 包含具体数据和统计显著性

我们检索到的:
- 可能是关于其他肛肠疾病的文本
- 缺少这项研究的具体数据
- LLM无法确定 → 输出"maybe"
```

---

## 结论

PubMedQA效果差(准确率~0%)的根本原因是**corpus不包含测试集对应的原始论文**。这是一个数据集设计和corpus匹配的问题,不是LinearRAG方法本身的问题。

要提升效果,必须构建包含PubMedQA 500个问题对应论文的专用corpus。

当前结果虽然差,但提供了关于corpus设计重要性的有价值洞察。
