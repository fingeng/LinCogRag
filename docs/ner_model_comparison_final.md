# NER 模型终极对比

## 测试结果总结

| 模型 | 覆盖率 | 解剖术语 | Subword问题 | 速度 | 推荐度 |
|------|--------|---------|------------|------|--------|
| **BC5CDR (spaCy)** | 30% | ❌ | ✅ 无 | ⚡⚡⚡ | ⭐⭐ |
| **CRAFT (spaCy)** | 10% | ❌ | ✅ 无 | ⚡⚡⚡ | ⭐ |
| **biomedical-ner-all (HF)** | 94.9% | ⚠️ 部分 | ❌ 有 | ⚡ | ⭐⭐⭐ |
| **Enhanced (BC5CDR + Dict)** | 预计85% | ✅ | ✅ 无 | ⚡⚡⚡ | ⭐⭐⭐⭐⭐ |

## 详细分析

### BC5CDR + Medical Dictionary (最推荐)

```python
优点:
  ✅ 无 subword 问题
  ✅ 速度快 (spaCy)
  ✅ 可控性强 (字典可调整)
  ✅ 专注疾病+化学物质+解剖术语

缺点:
  ⚠️ 需要维护医学词典
  ⚠️ 词典覆盖不全

预期效果:
  - Entity Coverage: 85%
  - LLM Accuracy: 40-50%
  - 无 subword 切分问题
```

### biomedical-ner-all (备选)

```python
优点:
  ✅ 覆盖率最高 (94.9%)
  ✅ 实体类型丰富 (37种)
  ✅ 能识别大部分医学术语

缺点:
  ❌ Subword 切分问题严重
  ❌ 速度慢 (Transformer)
  ❌ 内存占用大

实际效果:
  - 提取实体: "man", "##dible" ← 需要后处理合并
  - 某些专业术语仍识别不了
```

## 推荐方案

### 🥇 方案 1: Enhanced NER (BC5CDR + 扩展医学词典)

```python
# 优势
1. 无 subword 问题
2. 速度快
3. 可控性强

# 实现
- 基础: BC5CDR (疾病 + 化学物质)
- 增强: 医学词典 (解剖 + 生理)
- 后处理: 合并相邻医学术语
```

### 🥈 方案 2: HF NER + Post-processing

```python
# 优势
1. 覆盖率最高
2. 实体类型丰富

# 实现
- 使用 biomedical-ner-all
- 后处理合并 subword tokens
- 过滤低置信度实体
```

## 代码实现
