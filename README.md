# Data Harness

A toolkit for large-scale LLM training data generation. It provides a thin, reusable inference engine layer and a growing collection of task-specific data recipes on top of it.

---

## Design Philosophy

**Separate "how to run inference" from "what to do with it."**

The `data-harness/` core handles the mechanics of calling a model efficiently — connection pooling, concurrency, retry logic, resume, progress tracking. Recipes in `recipe/` focus entirely on the data transformation logic. This keeps both layers simple and independently reusable.

Two inference modes are provided:

| Mode | When to use |
|------|-------------|
| **Offline** (`offline.py`) | You have GPU access and want to run a local model via SGLang for maximum throughput |
| **Online** (`online.py`) | You call a remote OpenAI-compatible API endpoint |

Both expose the same mental model: feed in a JSONL file, get back a JSONL file with a `response` field appended to each record.

---

## Repository Layout

```
data-harness/
├── data-harness/               # Core inference engines
│   ├── base.py                 # SGLang engine lifecycle & safe generation
│   ├── offline.py              # BatchInferenceEngine (local, high-throughput)
│   └── online.py               # OnlineBatchInferenceEngine (API-based)
│
└── recipe/                     # Data pipelines built on the engines
    ├── thinking_to_cot/        # Convert thinking-style answers to CoT format
    └── llm_judge/              # Score model outputs with an LLM judge
```

---

## Core Engines

### Offline — `BatchInferenceEngine`

Runs a local model with SGLang. Designed for high-throughput batch jobs on GPU clusters.

```python
from data_harness.offline import BatchInferenceEngine

async with BatchInferenceEngine(model_path="/path/to/model", tp_size=8, dp_size=8) as engine:
    await engine.run(
        input_file="input.jsonl",
        output_file="output.jsonl",
        sampling_params={"temperature": 1.0, "max_new_tokens": 32768},
        resume=True,   # skip already-completed IDs
    )
```

**Input format** — each JSONL line must have an `id` and a `prompt` field:
```json
{"id": "0001", "prompt": "What is the capital of France?"}
```

**Output** — the same object with `response` appended:
```json
{"id": "0001", "prompt": "...", "response": "Paris."}
```

### Online — `OnlineBatchInferenceEngine`

Calls any OpenAI-compatible `/chat/completions` endpoint. Supports both `prompt` strings and pre-built `messages` arrays as input.

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

Or use it directly from the command line:

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

**Key features shared by both engines:**
- Resume from checkpoint — already-written IDs are skipped automatically
- Configurable concurrency / in-flight requests
- Automatic retry with exponential back-off on transient errors

---

## Recipes

### `thinking_to_cot` — Convert Thinking to CoT

Turns a raw dataset with human/assistant conversations into a CoT-style training set using a two-stage pipeline:

```
Stage 1  →  Feed questions to a thinking model (e.g. DeepSeek-V3)
             to generate detailed thinking-style answers.

Stage 2  →  Rewrite those thinking answers into structured CoT format
             (# Step 1 / # Step 2 / ... / final answer) using another model.
```

The pipeline is designed for distributed execution on Kubernetes: split data into shards, dispatch one shard per pod, then merge.

```bash
# 1. Split into shards
bash recipe/thinking_to_cot/stage1_prepare_split.sh

# 2. Dispatch Stage 1 across K8s pods
bash recipe/thinking_to_cot/stage1_dispatch.sh

# 3. Merge Stage 1 outputs, split for Stage 2
bash recipe/thinking_to_cot/stage2_prepare_split.sh

# 4. Dispatch Stage 2 across K8s pods
bash recipe/thinking_to_cot/stage2_dispatch.sh
```

For a single-machine run, call `stage1_run_shard.py` / `stage2_run_shard.py` directly.

---

### `llm_judge` — Score Outputs with an LLM

Uses a large judge model to score multiple candidate outputs for the same question on four dimensions: **correctness (50)**, **logic (25)**, **clarity (15)**, **completeness (10)**.

**Input:**
```json
{
  "conversations": [{"from": "human", "value": "..."}, {"from": "gpt", "value": "reference answer"}],
  "JoyAI_output_0": "candidate answer 1",
  "JoyAI_output_1": "candidate answer 2"
}
```

**Run the full pipeline:**
```bash
python recipe/llm_judge/run_judge_pipeline.py \
  --input input.jsonl \
  --output-dir ./judge_output \
  --judge-model /path/to/judge/model
```

**Output** (`judge_scores.jsonl`):
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

## Adding a New Recipe

1. Create a directory under `recipe/your_recipe/`.
2. Write a `prepare_*.py` that reads raw data and emits `{"id": ..., "prompt": ...}` JSONL.
3. Call `BatchInferenceEngine` or `OnlineBatchInferenceEngine` to run inference.
4. Write a `merge_*.py` that joins inference output back to the source data.

The engines do not care what the prompts contain — all domain logic lives in the recipe.
