## 1. 参数解析 (parse_arguments)
- spacy_model: NER 模型 (默认: en_core_sci_scibert)
- embedding_model: 句子嵌入模型路径
- dataset_name: 数据集名称 (pubmed/medical)
- llm_model: 大语言模型 (gpt-4o-mini)
- max_workers: 并行处理线程数
- use_mirage: 是否使用 MIRAGE 基准测试
- mirage_dataset: MIRAGE 数据集类型
- chunks_limit: 限制加载的文档数量
