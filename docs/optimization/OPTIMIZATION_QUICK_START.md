# 优化指南 - 快速入口

> 本文档是 LinearRAG 医疗领域实现的优化指南快速入口

## 🚨 当前状态

- ⚠️ **检索速度**: 90秒/问题 (严重过慢)
- ⚠️ **预计完成时间**: 31+ 小时
- ✅ **方法正确**: 核心算法实现无误
- 🎯 **优化目标**: 提速至 5-10秒/问题

---

## ⚡ 5分钟快速优化 (推荐先做这个!)

**预期效果**: 3-5倍提速 (90秒 → 20-30秒)

### 步骤1: 停止当前运行
```bash
kill $(pgrep -f "run.py")
# 或者: kill 3478849
```

### 步骤2: 修改配置文件
打开 `src/config.py`，找到这3行并修改:

```python
# 第38行附近
max_iterations=2,  # 改为2 (原值3)

# 第39行附近  
iteration_threshold=0.3,  # 改为0.3 (原值0.1)

# 第40行附近
top_k_sentence=5,  # 改为5 (原值3)
```

### 步骤3: 重新运行 (先测试100个问题)
```bash
python run.py \
    --use_hf_ner \
    --embedding_model model/all-mpnet-base-v2 \
    --dataset_name pubmed \
    --llm_model gpt-4o-mini \
    --max_workers 8 \
    --use_mirage \
    --mirage_dataset medqa \
    --chunks_limit 10000 \
    --questions_limit 100 \
    > medqa_optimized_test.log 2>&1 &
```

### 步骤4: 监控速度
```bash
tail -f medqa_optimized_test.log | grep "Retrieving:"
```

应该看到类似:
```
Retrieving:  10%|█         | 10/100 [03:20<30:00, 20.0s/it]  # ✅ 20秒/问题
```

如果速度正常,运行完整测试:
```bash
kill $(pgrep -f "run.py")
python run.py ... --questions_limit 1273 > medqa_optimized_full.log 2>&1 &
```

---

## 📚 详细文档

### 1. 综合分析报告 (必读!)
**文件**: `OPTIMIZATION_SUMMARY.md`

内容:
- ✅ 方法正确性验证
- ⚠️ 性能瓶颈分析  
- 🔧 3种优化方案 (快速/中等/完全)
- 📊 性能对比表
- 🎯 立即行动指南

### 2. 深度技术分析
**文件**: `docs/medical_linearrag_analysis.md`

内容:
- 核心代码逐行分析
- 图规模与效率矛盾
- NER策略冗余性分析
- 检索流程优化建议
- 相关工作对比

### 3. 优化代码实现
**文件**: `advanced_optimization_code.py`

内容:
- 5个可直接使用的优化函数
- 候选集预筛选实现
- Early stopping 实现
- 简化NER策略代码
- 性能对比表

### 4. 自动化工具

#### 工具A: 快速分析器
```bash
python quick_optimize.py
```
功能: 分析当前进度, 生成优化配置, 提供建议

#### 工具B: 测试脚本
```bash
./test_optimization.sh
```
功能: 自动测试多种配置, 对比结果

---

## 🎯 优化路线图

```
┌─────────────────────────────────────────────────────────────┐
│                     优化路线图                               │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  今天 (5分钟)                                                │
│  ├─ 停止当前运行                                             │
│  ├─ 修改3个参数                                              │
│  ├─ 测试100个问题                                            │
│  └─ 验证速度 (应该20-30秒)                                   │
│      ↓                                                       │
│  明天 (2小时) - 如果效果好                                    │
│  ├─ 实现候选集预筛选                                         │
│  ├─ 测试100个问题                                            │
│  └─ 对比准确率和速度                                         │
│      ↓                                                       │
│  下周 (1天) - 如果需要                                        │
│  ├─ 简化NER策略                                              │
│  ├─ 添加缓存机制                                             │
│  ├─ 完整测试1273问题                                         │
│  └─ 撰写实验报告                                             │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

---

## 📊 预期性能对比

| 阶段 | 速度 | 总时间 | 准确率 | 难度 |
|------|------|--------|--------|------|
| 当前 | 90秒/问题 | 53小时 | 基线 | - |
| 今天 (参数优化) | 20-30秒 | 7-10小时 | -1% | ⭐ |
| 明天 (候选集) | 7-11秒 | 2.5-4小时 | -1.5% | ⭐⭐ |
| 下周 (完全优化) | 4-6秒 | 1.5-2小时 | -2.5% | ⭐⭐⭐ |

---

## ❓ 常见问题

### Q1: 为什么速度这么慢?
**A**: 最大问题是在全图(21万节点)上运行PageRank,没有候选集预筛选。

### Q2: 会不会影响准确率?
**A**: 
- 快速优化 (方案1): 准确率下降约1%
- 如果下降<2%, 速度提升>3x, 就非常值得

### Q3: 需要修改很多代码吗?
**A**: 
- 快速优化: 只改3个参数,5分钟完成
- 完整优化: 需要1-2天

### Q4: 如何验证优化效果?
**A**: 
1. 先用100个问题测试
2. 对比速度 (应该明显变快)
3. 对比准确率 (最终运行完看结果)
4. 如果速度提升>3x, 准确率下降<2%, 就继续

### Q5: 如果优化后效果不好怎么办?
**A**: 
1. 所有配置都有备份 (config_backup_*.py)
2. 可以随时恢复: `cp src/config_backup.py src/config.py`
3. 建议先测试100个问题,确认OK再运行全部

---

## 🔍 关键指标监控

### 检索速度
```bash
# 实时监控
tail -f medqa_optimized_test.log | grep "Retrieving:"

# 统计平均值
grep "Retrieving:" medqa_optimized_test.log | grep -oP "\d+\.\d+s/it" | \
    awk '{sum+=$1; count++} END {print "平均:", sum/count, "秒/问题"}'
```

### 图规模
```bash
grep -E "Entity embeddings:|Passage embeddings:" medqa_optimized_test.log
```

应该看到:
```
Entity embeddings: (212532, 768)  # 21万实体
Passage embeddings: (49999, 768)  # 5万文档
```

### 准确率
```bash
# 等运行完成后
grep "Accuracy:" medqa_optimized_test.log
```

---

## 🚀 立即开始

**推荐步骤** (最快路径):

1. ✅ 阅读本文档 (你在这里!)
2. ⚡ 执行5分钟快速优化 (见上方)
3. 📊 监控速度改善
4. 📚 如果需要进一步优化,阅读详细文档

**关键文件**:
- 快速入口: 本文档
- 详细指南: `OPTIMIZATION_SUMMARY.md`
- 技术分析: `docs/medical_linearrag_analysis.md`
- 优化代码: `advanced_optimization_code.py`

---

## 📞 获取帮助

- 查看文档: 所有优化文档都在项目根目录
- 运行工具: `python quick_optimize.py`
- 测试脚本: `./test_optimization.sh`

---

**记住**: 优化是渐进的过程,先做简单的(5分钟参数调整),验证效果好再继续! 🎯
