# NER 模型最终推荐

## 实验结果总结

| 模型 | 识别能力 | 状态 | 推荐度 |
|------|---------|------|--------|
| **BC5CDR** | 疾病+化学物质 | ✅ 稳定 | ⭐⭐⭐⭐ |
| **CRAFT** | 化学物质(有限) | ⚠️ 解剖术语差 | ⭐⭐ |
| **HF biomedical-ner-all** | 全面 | ❌ 网络问题 | ⭐⭐⭐ |

## 最终推荐：BC5CDR + 关键词扩展

由于:
1. CRAFT 无法识别解剖术语
2. HF 模型下载困难
3. BC5CDR 稳定但覆盖有限

**解决方案**: BC5CDR + 医学词典增强

```python
# 在 BC5CDR 基础上添加医学术语词典
MEDICAL_TERMS = {
    # 解剖结构
    'anatomy': [
        'kidney', 'nephron', 'glomerulus', 'capillary',
        'atrium', 'ventricle', 'sinoatrial', 'node',
        'esophagus', 'sphincter', 'femur', 'patella',
        'ectomesenchyme', 'mesenchyme', 'vertebra'
    ],
    # 生理过程
    'physiology': [
        'filtration', 'secretion', 'absorption', 'metabolism',
        'circulation', 'respiration', 'digestion'
    ],
    # 症状
    'symptoms': [
        'pain', 'fever', 'dyspnea', 'reflux', 'compression'
    ]
}
```
