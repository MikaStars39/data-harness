#!/bin/bash
# LLM Judge Scoring System - Example Run Script

# Configuration paths
INPUT_FILE="/mnt/llm-train/users/explore-train/wangzhenfang8/codes/generate/data/math500_and_gsm8k/math500_and_gsm8k_roll_8.jsonl"
OUTPUT_DIR="/mnt/llm-train/users/explore-train/qingyu/data/dpo/yuqi_math_20k"
JUDGE_MODEL="/mnt/llm-train/users/explore-train/qingyu/.cache/Qwen3-30B-A3B-Thinking-2507"
SCRIPT_DIR="/mnt/llm-train/users/explore-train/qingyu/slimulation"

# 1. 定义 Pod 列表 (按顺序排列，共16个)
pods=(
    "dpo-data-0-6vd8m"
    "dpo-data-1-qpcgs"
    "dpo-data-2-sk8rp"
    "dpo-data-3-gj6wd"
    "dpo-data-4-hl59h"
    "dpo-data-5-rp6nz"
    "dpo-data-6-blg5d"
    "dpo-data-7-ch5qw"
)
# for pod in "${pods[@]}"; do echo "Cleaning $pod..."; kt exec $pod -- pkill -9 -f sglang || echo "No sglang process found on $pod"; done

# 2. 循环启动任务
for i in "${!pods[@]}"; do
    pod_name=${pods[$i]}
    shard_id=$i
    
    echo "正在为 Pod [$pod_name] 分配任务: Shard $shard_id ..."

    # 使用 nohup 或后台运行模式执行，避免 kubectl 断开导致任务终止
    kt exec $pod_name -- bash -c "pkill sglang && python $SCRIPT_DIR/recipe/llm_judge/inference.py \
        --input \"$OUTPUT_DIR/shards_16_yuqi_new_data/shard_${shard_id}.jsonl\" \
        --output \"$OUTPUT_DIR/responses/response_${shard_id}.jsonl\" \
        --model_path $JUDGE_MODEL \
        --tp_size 1 \
        --dp_size 8 \
        --max_tokens 32768
    " > "log_shard_${shard_id}.log" 2>&1 &

    # 稍微等一秒，避免同时并发请求 kubectl API 过载
    sleep 1
done

python $SCRIPT_DIR/recipe/llm_judge/merge_and_extract.py \
    --response-dir $OUTPUT_DIR/responses \
    --output $OUTPUT_DIR/scores.jsonl \
    --failed $OUTPUT_DIR/failed.jsonl \
    --workers 48

python $SCRIPT_DIR/recipe/llm_judge/analyze_best_worst.py \
    --input $OUTPUT_DIR/scores.jsonl \
    --output $OUTPUT_DIR/final.jsonl

python $SCRIPT_DIR/recipe/llm_judge/analyze_scores.py \
    --scores $OUTPUT_DIR/final.jsonl