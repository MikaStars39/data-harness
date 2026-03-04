#!/usr/bin/env bash
set -euo pipefail

# Usage:
# 1) Edit RAW_SOURCE_JSONL / OUTPUT_DIR / MODEL_PATH below.
# 2) bash recipe/thinking_to_cot/run_deepseek_v32_convert.sh
export FLASHINFER_DISABLE_VERSION_CHECK=1

RAW_SOURCE_JSONL=/jpfs/qingyu/data/all.jsonl
OUTPUT_DIR=/jpfs/qingyu/data/all_output.jsonl
MODEL_PATH=/jpfs/models/DeepSeek-V3.2

ENGINE_TYPE="online"
API_KEY="your-api-key-here"
BASE_URL="http://11.48.241.174:30000/v1"
CONNECTOR_LIMIT=0
LIMIT_PER_HOST=0
WRITER_FLUSH_EVERY=256
USE_CHAT_TEMPLATE="false"
PREPROCESS_WORKERS=8
PREPROCESS_CHUNKSIZE=512
SKIP_PREPARE_ON_RESUME="true"

mkdir -p "$OUTPUT_DIR"

SOURCE_JSONL="$OUTPUT_DIR/source.normalized_for_thinking_to_cot.jsonl"
PREPARED_JSONL="$OUTPUT_DIR/thinking_to_cot.prepared.jsonl"
MODEL_OUTPUT_JSONL="$OUTPUT_DIR/thinking_to_cot.model_output.jsonl"
FINAL_JSONL="$OUTPUT_DIR/thinking_to_cot.final.jsonl"

if [[ "$SKIP_PREPARE_ON_RESUME" != "true" ]]; then
  python recipe/thinking_to_cot/preprocess_conversations_to_thinking.py \
    --raw-jsonl "$RAW_SOURCE_JSONL" \
    --normalized-jsonl "$SOURCE_JSONL" \
    --workers "$PREPROCESS_WORKERS" \
    --chunksize "$PREPROCESS_CHUNKSIZE"
fi

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
  --max-new-tokens 65536 \
  --engine-type "$ENGINE_TYPE" \
  --api-key "$API_KEY" \
  --base-url "$BASE_URL" \
  --connector-limit "$CONNECTOR_LIMIT" \
  --limit-per-host "$LIMIT_PER_HOST" \
  --writer-flush-every "$WRITER_FLUSH_EVERY" \
  $([[ "$SKIP_PREPARE_ON_RESUME" == "true" ]] && echo "--skip-prepare" || echo "") \
  $([[ "$USE_CHAT_TEMPLATE" == "true" ]] && echo "--apply-chat-template" || echo "--no-apply-chat-template") \
  --resume

echo "Done. Final output: $FINAL_JSONL"
