#!/bin/bash
# LLM Judge Scoring System - Example Run Script

# Configuration paths
INPUT_FILE="/jpfs/qingyu/data/deploy-sft-128k-s2-0318-s2-fixswe-shuf-64-1e-5-min1e-6-outputs_tmp.jsonl"
OUTPUT_DIR="/jpfs/qingyu/data-harness/output/dpo_data_0319"
JUDGE_MODEL="/jpfs/models/DeepSeek-V3.2"
SCRIPT_DIR="/jpfs/qingyu/data-harness"

python $SCRIPT_DIR/recipe/llm_judge/prepare_judge.py \
    --input $INPUT_FILE \
    --output "$OUTPUT_DIR/prepare.jsonl" \
    --tokenizer $JUDGE_MODEL \
    --workers 128

python $SCRIPT_DIR/recipe/llm_judge/shard_jsonl.py \
    --input $OUTPUT_DIR/prepare.jsonl \
    --output-dir "$OUTPUT_DIR/shards_233_yuqi_new_data" \
    --num-shards 233