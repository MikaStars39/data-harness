#!/usr/bin/env bash
set -euo pipefail

# Usage:
# 1) Edit RAW_SOURCE_JSONL / OUTPUT_DIR / MODEL_PATH below.
# 2) bash recipe/thinking_to_cot/run_deepseek_v32_convert.sh
export FLASHINFER_DISABLE_VERSION_CHECK=1

RAW_SOURCE_JSONL=/mnt/llm-train/users/explore-train/qingyu/data/251230-nankai-filtered_16384_100.jsonl
OUTPUT_DIR=/mnt/llm-train/users/explore-train/qingyu/data-harness/output/251230-nankai-filtered_16384/
MODEL_PATH=/mnt/llm-train/users/explore-train/qingyu/.cache/DeepSeek-V3.2/DeepSeek-V3.2

ENGINE_TYPE="online"
API_KEY="your-api-key-here"
BASE_URL="your-base-url-here"

mkdir -p "$OUTPUT_DIR"

SOURCE_JSONL="$OUTPUT_DIR/source.normalized_for_thinking_to_cot.jsonl"
PREPARED_JSONL="$OUTPUT_DIR/thinking_to_cot.prepared.jsonl"
MODEL_OUTPUT_JSONL="$OUTPUT_DIR/thinking_to_cot.model_output.jsonl"
FINAL_JSONL="$OUTPUT_DIR/thinking_to_cot.final.jsonl"

python recipe/thinking_to_cot/preprocess_conversations_to_thinking.py \
  --raw-jsonl "$RAW_SOURCE_JSONL" \
  --normalized-jsonl "$SOURCE_JSONL"

python recipe/thinking_to_cot/deepseek_v32_convert.py \
  --source-jsonl "$SOURCE_JSONL" \
  --prepared-jsonl "$PREPARED_JSONL" \
  --model-output-jsonl "$MODEL_OUTPUT_JSONL" \
  --final-jsonl "$FINAL_JSONL" \
  --model-path "$MODEL_PATH" \
  --tp-size 8 \
  --dp-size 1 \
  --enable-dp-attention \
  --max-inflight 1024 \
  --temperature 1 \
  --top-p 0.95 \
  --max-new-tokens 100000 \
  --apply-chat-template \
  --engine-type "$ENGINE_TYPE" \
  --api-key "$API_KEY" \
  --base-url "$BASE_URL" \
  --resume

echo "Done. Final output: $FINAL_JSONL"
