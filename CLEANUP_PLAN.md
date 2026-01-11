# LinearRAG 代码清理计划

## 📋 核心架构（必须保留）

### 主入口
- ✅ `run.py` - 主入口（CLI方式）
- ✅ `experiments/run_lincog_benchmark.py` - LinCog实验入口
- ✅ `requirements.txt` - 依赖

### 核心源码 (src/)
- ✅ `src/LinearRAG.py` - 核心算法实现（含Hypergraph）
- ✅ `src/config.py` - 配置类
- ✅ `src/ner.py` - NER实现（混合BC5CDR+HF）
- ✅ `src/embedding_store.py` - Embedding存储
- ✅ `src/llm.py` - LLM接口
- ✅ `src/dataset_loader.py` - 数据加载器
- ✅ `src/cli.py` - 命令行参数解析
- ✅ `src/pipeline.py` - 主流程
- ✅ `src/utils.py` - 工具函数
- ✅ `src/eval/` - 评估模块

### Hypergraph模块 (src/hypergraph/)
- ✅ `src/hypergraph/__init__.py`
- ✅ `src/hypergraph/cooccurrence_hyperedge.py` - 超边构建+医学增强
- ✅ `src/hypergraph/hypergraph_store.py` - 超图存储
- ✅ `src/hypergraph/cache_manager.py` - 缓存管理
- ✅ `src/hypergraph/incremental_index.py` - 增量索引

### 数据和结果（必须保留）
- ✅ `dataset/` - 原始数据集
- ✅ `MIRAGE/` - MIRAGE基准数据
- ✅ `artifacts/lincog_benchmark/` - 实验结果
- ✅ `import/` - 索引缓存
- ✅ `model/` 或 `models/` - 预训练模型

### 文档
- ✅ `README.md`
- ✅ `LinearRAG完整流程解析.md` - 新增的详细文档

---

## 🗑️ 可以删除的文件

### 1. 冗余NER实现（已合并到src/ner.py）
- ❌ `src/ner_enhanced.py` - 功能已集成到ner.py的医学模式中
- ❌ `src/ner_huggingface.py` - 功能已集成到ner.py的混合NER中

### 2. 测试脚本 (scripts/test_*.py)
❌ 以下测试脚本可删除（功能已验证）：
- `scripts/test_craft_ner.py`
- `scripts/test_downloaded_biomedical_ner.py`
- `scripts/test_enhanced_ner.py`
- `scripts/test_hf_ner.py`
- `scripts/test_ner_simple.py`
- `scripts/test_pubmedqa_no_rag.py`
- `scripts/test_pubmedqa_with_context.py`
- `scripts/test_pubmedqa_with_context_in_graph.py`
- `scripts/test_sentence_extraction.py`
- `scripts/test_subword_merge.py`

### 3. 工具测试 (tools/tests/)
❌ 以下测试文件可删除：
- `tools/tests/test_biomedical_ner_local.py`
- `tools/tests/test_enhanced_ner_standalone.py`
- `tools/tests/test_hf_ner.py`
- `tools/tests/test_medmcqa_loading.py`
- `tools/tests/test_ner.py`
- `tools/tests/test_single_question.py`
- `tools/tests/test_spacy_load.py`

### 4. 辅助脚本（根据使用情况保留部分）
❌ 可删除的诊断脚本：
- `scripts/diagnose_mirage.py` - 一次性诊断脚本
- `scripts/diagnose_results.py` - 一次性诊断脚本
- `scripts/check_gpu.py` - 简单的GPU检查
- `scripts/check_gpu_memory.py` - GPU内存检查

✅ 保留有用的工具：
- `scripts/analyze_graph.py` - 图分析工具
- `scripts/visualize_results.py` - 结果可视化
- `scripts/download_biomedical_ner.py` - 模型下载
- `scripts/evaluate_hyperlinearrag.py` - 评估脚本

### 5. 实验脚本（保留核心的）
✅ 保留：
- `experiments/run_lincog_benchmark.py` - 主实验
- `experiments/run_benchmark.sh` - Shell入口

❌ 可删除：
- `scripts/run_hyperlinearrag_test.py` - 重复的测试
- `scripts/run_hyperlinearrag_test.sh`
- `scripts/start_hyperlinearrag_test.sh`
- `scripts/start_pubmedqa_context_test.sh`

### 6. 旧的/重复的Shell脚本
❌ 以下可删除或整合：
- `scripts/quick_test.sh` - 一次性测试
- `scripts/switch_to_bc5cdr.sh` - 一次性切换
- `scripts/download_hf_model.sh` - 可用download_biomedical_ner.py替代

### 7. 重复的配置目录
检查后：
- ✅ `config/medical_terms.json` - 医学术语（如果ner.py中没用到可删）
- ✅ `configs/ner_models.yaml` - NER模型配置（如果未使用可删）

---

## 📊 清理统计

### 预估可删除文件数
- NER实现: 2个
- 测试脚本: ~18个
- 辅助/诊断脚本: ~10个
- 重复Shell: ~5个
- **总计: ~35个文件**

### 预估保留文件
- 核心源码: ~15个
- Hypergraph模块: 4个
- 实验脚本: 2个
- 有用工具: ~5个
- 数据/结果: 保留所有目录

---

## ⚠️ 清理前检查清单

### 必须确认的引用
1. ✅ `ner_enhanced.py` 和 `ner_huggingface.py` 没有被主流程引用
2. ✅ 测试脚本不包含实验结果或重要配置
3. ✅ 删除前备份 `artifacts/` 目录

### 保留原则
1. 任何包含实验结果的文件
2. README或文档引用的脚本
3. 依赖关系不明确的文件（先标记，后续确认）

---

## 🔄 清理步骤

### Phase 1: 安全删除（无依赖）
1. 删除明确的测试脚本
2. 删除诊断/一次性脚本

### Phase 2: 确认后删除
1. 检查NER文件是否被引用
2. 删除冗余NER实现

### Phase 3: 整理配置
1. 合并或删除重复配置
2. 更新README（如有必要）

### Phase 4: 验证
1. 运行主流程确认无错误
2. 检查实验脚本可用性


