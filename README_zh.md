# Data Harness

用于大规模 LLM 训练数据生成的工具框架。提供一个轻量、可复用的推理引擎层，以及在此之上构建的各类数据处理 recipe。

---

## 设计哲学

**把"怎么调模型"和"数据怎么处理"彻底分开。**

`data-harness/` 核心层只负责高效调用模型这件事——连接池、并发控制、重试、断点续传、进度跟踪。`recipe/` 层只负责数据变换逻辑，完全不关心底层怎么跑推理。两层各自简单，也各自独立可复用。

两种推理模式：

| 模式 | 适用场景 |
|------|----------|
| **离线** (`offline.py`) | 有 GPU 资源，用 SGLang 跑本地模型，追求极致吞吐 |
| **在线** (`online.py`) | 调用远程 OpenAI 兼容 API |

两者对外接口一致：输入一个 JSONL 文件，输出每条记录追加了 `response` 字段的 JSONL 文件。

---

## 项目结构

```
data-harness/
├── data-harness/               # 推理引擎核心
│   ├── base.py                 # SGLang 引擎生命周期管理 & 安全生成
│   ├── offline.py              # BatchInferenceEngine（本地高吞吐）
│   └── online.py               # OnlineBatchInferenceEngine（API 调用）
│
└── recipe/                     # 基于引擎构建的数据流水线
    ├── thinking_to_cot/        # Thinking 风格答案 → CoT 格式转换
    └── llm_judge/              # 用 LLM 对多个模型输出打分
```

---

## 推理引擎

### 离线模式 — `BatchInferenceEngine`

在本地用 SGLang 加载模型，适合 GPU 集群上的大批量任务。

```python
from data_harness.offline import BatchInferenceEngine

async with BatchInferenceEngine(model_path="/path/to/model", tp_size=8, dp_size=8) as engine:
    await engine.run(
        input_file="input.jsonl",
        output_file="output.jsonl",
        sampling_params={"temperature": 1.0, "max_new_tokens": 32768},
        resume=True,   # 自动跳过已完成的 ID，支持断点续传
    )
```

**输入格式** — 每行 JSON 必须有 `id` 和 `prompt`：
```json
{"id": "0001", "prompt": "法国的首都是哪里？"}
```

**输出** — 原始对象追加 `response`：
```json
{"id": "0001", "prompt": "...", "response": "巴黎。"}
```

### 在线模式 — `OnlineBatchInferenceEngine`

调用任意 OpenAI 兼容的 `/chat/completions` 接口。输入支持 `prompt` 字符串或完整的 `messages` 数组。

```python
from data_harness.online import APIConfig, OnlineBatchInferenceEngine

config = APIConfig(api_key="sk-...", base_url="https://api.openai.com/v1", model="gpt-4o")
engine = OnlineBatchInferenceEngine(config, concurrency=100)

await engine.run(
    input_file="input.jsonl",
    output_file="output.jsonl",
    sampling_params={"temperature": 0.7, "max_tokens": 2048},
)
```

也可以直接命令行调用：

```bash
python -m data_harness.online \
  --api-key sk-... \
  --base-url https://api.openai.com/v1 \
  --model gpt-4o \
  --input input.jsonl \
  --output output.jsonl \
  --concurrency 100 \
  --temperature 0.7 \
  --max-tokens 2048
```

**两种引擎共有能力：**
- 断点续传 — 已写入的 ID 自动跳过
- 并发 / 在途请求数可配置
- 遇到网络或服务错误自动指数退避重试

---

## Recipe

### `thinking_to_cot` — Thinking 转 CoT

将原始对话数据集转换为结构化 CoT 格式的训练集，分两个阶段：

```
Stage 1  →  把问题喂给 thinking 模型（如 DeepSeek-V3）
             生成详细的 thinking 风格回答

Stage 2  →  把 thinking 回答 rewrite 成结构化 CoT 格式
             （# Step 1 / # Step 2 / ... / 最终答案）
```

流水线专为 Kubernetes 分布式执行设计：把数据 split 成 shard，每个 pod 跑一个 shard，最后 merge。

```bash
# 1. 切分 shard
bash recipe/thinking_to_cot/stage1_prepare_split.sh

# 2. Stage 1 分发到 K8s pods
bash recipe/thinking_to_cot/stage1_dispatch.sh

# 3. 合并 Stage 1 输出，切分 Stage 2 输入
bash recipe/thinking_to_cot/stage2_prepare_split.sh

# 4. Stage 2 分发到 K8s pods
bash recipe/thinking_to_cot/stage2_dispatch.sh
```

单机运行直接调用 `stage1_run_shard.py` / `stage2_run_shard.py` 即可。

---

### `llm_judge` — LLM 打分

用大模型对同一问题的多个候选回答进行多维度评分：**正确性（50分）**、**逻辑性（25分）**、**清晰度（15分）**、**完整性（10分）**，满分 100 分。

**输入格式：**
```json
{
  "conversations": [{"from": "human", "value": "..."}, {"from": "gpt", "value": "参考答案"}],
  "JoyAI_output_0": "候选答案1",
  "JoyAI_output_1": "候选答案2"
}
```

**一键运行完整流程：**
```bash
python recipe/llm_judge/run_judge_pipeline.py \
  --input input.jsonl \
  --output-dir ./judge_output \
  --judge-model /path/to/judge/model
```

**输出** (`judge_scores.jsonl`)：
```json
{
  "question": "...",
  "reference_answer": "...",
  "scores": {
    "JoyAI_output_0": {"correctness": 45, "logic": 22, "clarity": 13, "completeness": 9, "total_score": 89}
  }
}
```

---

## 新增 Recipe

1. 在 `recipe/your_recipe/` 下建目录。
2. 写 `prepare_*.py`，读入原始数据，输出 `{"id": ..., "prompt": ...}` 格式的 JSONL。
3. 调用 `BatchInferenceEngine` 或 `OnlineBatchInferenceEngine` 跑推理。
4. 写 `merge_*.py`，把推理输出和原始数据合并成最终交付格式。

引擎不关心 prompt 内容——所有领域逻辑都在 recipe 层。
