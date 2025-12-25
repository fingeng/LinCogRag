## 目标

你提出的两个核心诉求：

- **工程化**：把 `run.py` 拆成稳定模块、把 5 个问答集（`medqa/medmcqa/mmlu/pubmedqa/bioasq`）提供统一接口；
- **仓库清理**：把日志/结果/一次性脚本从根目录移走，保留“可复现、可复盘”的最小集合。

本次改造遵循原则：**不破坏已有索引复用能力**（`import/<dataset_namespace>/` 下的 parquet/NER/graphml）。

---

## 新的执行入口

仍然使用根目录 `run.py`，但它现在只做两件事：

1. 解析 CLI 参数（`src/cli.py`）
2. 调用统一管线（`src/pipeline.py`）

---

## 统一数据接口（5 个问答集）

统一 loader 位于：

- `src/dataset_loader.py`

核心函数：

- `load_mirage_questions_local(mirage_dataset, questions_limit=None, mirage_root="MIRAGE/rawdata")`
- `load_pubmed_passages(chunks_dir="dataset/pubmed/chunk", chunks_limit=None)`

返回的 question 统一为字典：

- `question`: str  
  - 对四选一（`medqa/medmcqa/mmlu`）会把选项拼进 question 文本中，便于 LLM 严格输出字母
- `answer`: str  
  - `medqa/medmcqa/mmlu`: `A/B/C/D`
  - `pubmedqa`: `yes/no/maybe`
  - `bioasq`: `yes/no`
- `dataset`: str

---

## 统一运行管线

主逻辑位于：

- `src/pipeline.py`

流程：

1. 加载 embedding 模型（SentenceTransformer）
2. 加载 passages（PubMed chunk）+ questions（MIRAGE 五数据集）
3. 选择索引命名空间 `import/<namespace>/`
4. `LinearRAG.index(passages)`（自动复用已有 parquet/NER/graphml）
5. `LinearRAG.qa(questions)`（按 dataset 类型强制输出格式）
6. `src/eval/summary.py` 做统一统计并保存

输出默认写到：

- `experiments/results/<run_name>.json`
- `experiments/results/<run_name>_summary.json`

---

## 评测统计（修复 pubmedqa/bioasq “全 invalid”）

之前根目录 `run.py` 的统计逻辑把所有非 `A/B/C/D` 都算 invalid，导致 `pubmedqa/bioasq` 的 invalid 数量恒等于总题数。

现在统一统计在：

- `src/eval/summary.py`

规则：

- `medqa/medmcqa/mmlu`: valid = `{A,B,C,D}`
- `pubmedqa`: valid = `{Yes,No,Maybe}`（大小写不敏感）
- `bioasq`: valid = `{Yes,No}`（大小写不敏感）

并且 `src/LinearRAG.py` 对 pubmedqa/bioasq 的解析失败会直接输出 `INVALID`，便于定位 prompt/解析问题。

---

## 常用命令（示例）

### 1) 跑 MedQA（复用/构建索引）

```bash
python run.py \
  --use_mirage \
  --mirage_dataset medqa \
  --embedding_model model/all-mpnet-base-v2 \
  --dataset_name pubmed \
  --llm_model gpt-4o-mini \
  --max_workers 8 \
  --chunks_limit 10000 \
  --questions_limit -1
```

### 2) 复用已有索引目录（强制使用 import/pubmed_mirage_medqa）

```bash
python run.py \
  --use_mirage \
  --mirage_dataset medmcqa \
  --reuse_index pubmed_mirage_medqa \
  --embedding_model model/all-mpnet-base-v2 \
  --dataset_name pubmed \
  --llm_model gpt-4o-mini \
  --max_workers 8 \
  --chunks_limit 10000
```

---

## 清理建议（安全优先）

强烈建议把根目录的 `*.log / *.pid / results_*.json` 等运行产物移动到 `experiments/` 下（而不是直接删除），避免丢失实验复盘信息。

同时已经新增根目录 `.gitignore`，默认不提交：

- `import/`（索引/缓存）
- `experiments/`（实验产物）
- `*.log / *.pid / nohup.out`


