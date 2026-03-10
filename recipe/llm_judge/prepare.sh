#!/bin/bash
# LLM Judge Scoring System - Example Run Script

# Configuration paths
INPUT_FILE="/mnt/llm-train/users/explore-train/wangzhenfang8/codes/generate/data/math500_and_gsm8k/math500_and_gsm8k_roll_8.jsonl"
OUTPUT_DIR="/mnt/llm-train/users/explore-train/qingyu/data/dpo/yuqi_math_20k"
JUDGE_MODEL="/mnt/llm-train/users/explore-train/qingyu/.cache/Qwen3-30B-A3B-Thinking-2507"
SCRIPT_DIR="/mnt/llm-train/users/explore-train/qingyu/slimulation"

python $SCRIPT_DIR/recipe/llm_judge/prepare_judge.py \
    --input $INPUT_FILE \
    --output "$OUTPUT_DIR/prepare.jsonl" \
    --tokenizer $JUDGE_MODEL

python $SCRIPT_DIR/recipe/llm_judge/shard_jsonl.py \
    --input $OUTPUT_DIR/prepare.jsonl \
    --output-dir "$OUTPUT_DIR/shards_16_yuqi_new_data" \
    --num-shards 16 \
    --num-readers 48