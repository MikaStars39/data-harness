#!/usr/bin/env bash
set -euo pipefail

# Node-level sharded offline pipeline.
# 1) Optionally run preprocess once (typically rank 0).
# 2) Build shard-local prepared file.
# 3) Run local model inference on this shard only.
# 4) Merge shard output to shard final JSONL.

RAW_SOURCE_JSONL=/jpfs/qingyu/data/all.jsonl
OUTPUT_DIR=/jpfs/qingyu/data/all_output.jsonl
MODEL_PATH=/jpfs/models/DeepSeek-V3.2

NUM_SHARDS=8
SHARD_RANK=0

# Set false after source/prepared are ready.
RUN_PREPROCESS="false"
PREPROCESS_WORKERS=8
PREPROCESS_CHUNKSIZE=512

TP_SIZE=8
DP_SIZE=1
MAX_INFLIGHT=1024
ENABLE_DP_ATTENTION="true"
TEMPERATURE=1
TOP_P=0.95
MAX_NEW_TOKENS=65536
RESUME="true"
SKIP_BUILD_SHARD_PREPARED="false"

mkdir -p "$OUTPUT_DIR"

SOURCE_JSONL="$OUTPUT_DIR/source.normalized_for_thinking_to_cot.jsonl"
PREPARED_JSONL="$OUTPUT_DIR/thinking_to_cot.prepared.jsonl"
SHARD_PREPARED_JSONL="$OUTPUT_DIR/thinking_to_cot.prepared.shard_${SHARD_RANK}_of_${NUM_SHARDS}.jsonl"
SHARD_MODEL_OUTPUT_JSONL="$OUTPUT_DIR/thinking_to_cot.model_output.shard_${SHARD_RANK}_of_${NUM_SHARDS}.jsonl"
SHARD_FINAL_JSONL="$OUTPUT_DIR/thinking_to_cot.final.shard_${SHARD_RANK}_of_${NUM_SHARDS}.jsonl"

if [[ "$RUN_PREPROCESS" == "true" ]]; then
  python recipe/thinking_to_cot/preprocess_conversations_to_thinking.py \
    --raw-jsonl "$RAW_SOURCE_JSONL" \
    --normalized-jsonl "$SOURCE_JSONL" \
    --workers "$PREPROCESS_WORKERS" \
    --chunksize "$PREPROCESS_CHUNKSIZE"

  # Reuse existing convert script only for fast prepare file generation.
  python recipe/thinking_to_cot/deepseek_v32_convert.py \
    --source-jsonl "$SOURCE_JSONL" \
    --prepared-jsonl "$PREPARED_JSONL" \
    --model-output-jsonl "$OUTPUT_DIR/thinking_to_cot.model_output.tmp.jsonl" \
    --final-jsonl "$OUTPUT_DIR/thinking_to_cot.final.tmp.jsonl" \
    --model-path "$MODEL_PATH" \
    --no-apply-chat-template \
    --prepare-only
fi

python recipe/thinking_to_cot/deepseek_v32_offline_shard.py \
  --source-jsonl "$SOURCE_JSONL" \
  --prepared-jsonl "$PREPARED_JSONL" \
  --shard-prepared-jsonl "$SHARD_PREPARED_JSONL" \
  --shard-model-output-jsonl "$SHARD_MODEL_OUTPUT_JSONL" \
  --shard-final-jsonl "$SHARD_FINAL_JSONL" \
  --num-shards "$NUM_SHARDS" \
  --shard-rank "$SHARD_RANK" \
  --model-path "$MODEL_PATH" \
  --tp-size "$TP_SIZE" \
  --dp-size "$DP_SIZE" \
  --max-inflight "$MAX_INFLIGHT" \
  --temperature "$TEMPERATURE" \
  --top-p "$TOP_P" \
  --max-new-tokens "$MAX_NEW_TOKENS" \
  $([[ "$ENABLE_DP_ATTENTION" == "true" ]] && echo "--enable-dp-attention" || echo "") \
  $([[ "$RESUME" == "true" ]] && echo "--resume" || echo "") \
  $([[ "$SKIP_BUILD_SHARD_PREPARED" == "true" ]] && echo "--skip-build-shard-prepared" || echo "")

echo "Done. Shard final output: $SHARD_FINAL_JSONL"
