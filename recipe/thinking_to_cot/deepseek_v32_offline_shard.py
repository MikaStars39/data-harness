import argparse
import asyncio
import hashlib
import importlib.util
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Set, Tuple


def _load_batch_inference_engine(repo_root: Path):
    package_name = "data_harness_runtime"
    package_dir = repo_root / "data-harness"
    base_path = package_dir / "base.py"
    offline_path = package_dir / "offline.py"

    if not base_path.exists() or not offline_path.exists():
        raise FileNotFoundError("Cannot find `data-harness/base.py` or `data-harness/offline.py`.")

    package_spec = importlib.util.spec_from_loader(package_name, loader=None)
    package_module = importlib.util.module_from_spec(package_spec)
    package_module.__path__ = [str(package_dir)]
    sys.modules[package_name] = package_module

    base_spec = importlib.util.spec_from_file_location(f"{package_name}.base", base_path)
    base_module = importlib.util.module_from_spec(base_spec)
    sys.modules[f"{package_name}.base"] = base_module
    assert base_spec.loader is not None
    base_spec.loader.exec_module(base_module)

    offline_spec = importlib.util.spec_from_file_location(f"{package_name}.offline", offline_path)
    offline_module = importlib.util.module_from_spec(offline_spec)
    sys.modules[f"{package_name}.offline"] = offline_module
    assert offline_spec.loader is not None
    offline_spec.loader.exec_module(offline_module)

    return offline_module.BatchInferenceEngine


def _in_shard(item_id: str, shard_rank: int, num_shards: int) -> bool:
    digest = hashlib.md5(item_id.encode("utf-8")).hexdigest()
    bucket = int(digest[:16], 16) % num_shards
    return bucket == shard_rank


def build_shard_prepared(prepared_jsonl: Path, shard_prepared_jsonl: Path, shard_rank: int, num_shards: int) -> Tuple[int, int]:
    shard_prepared_jsonl.parent.mkdir(parents=True, exist_ok=True)
    total = 0
    kept = 0
    with prepared_jsonl.open("r", encoding="utf-8") as fin, shard_prepared_jsonl.open("w", encoding="utf-8") as fout:
        for line in fin:
            if not line.strip():
                continue
            total += 1
            item = json.loads(line)
            item_id = str(item.get("id", ""))
            if not item_id:
                continue
            if not _in_shard(item_id, shard_rank=shard_rank, num_shards=num_shards):
                continue
            fout.write(json.dumps(item, ensure_ascii=False) + "\n")
            kept += 1
    return total, kept


def _parse_model_response(raw_text: str) -> Dict[str, str]:
    raw_text = (raw_text or "").strip()
    if not raw_text:
        return {"cot": "", "final_answer": ""}

    candidates = [raw_text]
    fenced_blocks = re.findall(r"```(?:json)?\s*(\{[\s\S]*?\})\s*```", raw_text, flags=re.IGNORECASE)
    candidates.extend(fenced_blocks)
    brace_start = raw_text.find("{")
    brace_end = raw_text.rfind("}")
    if brace_start != -1 and brace_end != -1 and brace_end > brace_start:
        candidates.append(raw_text[brace_start : brace_end + 1])

    for candidate in candidates:
        try:
            obj = json.loads(candidate)
            cot = str(obj.get("cot", "")).strip()
            final_answer = str(obj.get("final_answer", "")).strip()
            return {"cot": cot, "final_answer": final_answer}
        except json.JSONDecodeError:
            continue

    cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw_text, flags=re.IGNORECASE).strip()
    return {"cot": cleaned, "final_answer": ""}


def _compose_assistant_response(cot: str, final_answer: str) -> str:
    cot_text = (cot or "").strip()
    answer_text = (final_answer or "").strip()
    if cot_text and answer_text:
        return f"{cot_text}\n\n{answer_text}"
    return cot_text or answer_text


def _replace_last_assistant_message(conversations: Any, rewritten_text: str) -> list:
    conv_list = conversations if isinstance(conversations, list) else []
    copied = [dict(msg) if isinstance(msg, dict) else msg for msg in conv_list]
    for idx in range(len(copied) - 1, -1, -1):
        msg = copied[idx]
        if not isinstance(msg, dict):
            continue
        role = str(msg.get("from", "")).strip().lower()
        if role in {"gpt", "assistant"}:
            msg["value"] = rewritten_text
            return copied
    copied.append({"from": "gpt", "value": rewritten_text})
    return copied


def merge_shard_output(
    source_jsonl: Path,
    shard_prepared_jsonl: Path,
    shard_model_output_jsonl: Path,
    shard_final_jsonl: Path,
) -> int:
    shard_final_jsonl.parent.mkdir(parents=True, exist_ok=True)
    source_map: Dict[str, Dict[str, Any]] = {}
    prepared_meta: Dict[str, Dict[str, Any]] = {}
    written = 0

    with source_jsonl.open("r", encoding="utf-8") as fin:
        for idx, line in enumerate(fin):
            if not line.strip():
                continue
            item = json.loads(line)
            item_id = str(item.get("id", f"line-{idx}"))
            source_map[item_id] = item

    with shard_prepared_jsonl.open("r", encoding="utf-8") as fin:
        for line in fin:
            if not line.strip():
                continue
            item = json.loads(line)
            prepared_meta[str(item["id"])] = item.get("_meta", {})

    with shard_model_output_jsonl.open("r", encoding="utf-8") as fin, shard_final_jsonl.open("w", encoding="utf-8") as fout:
        for line in fin:
            if not line.strip():
                continue
            item = json.loads(line)
            item_id = str(item.get("id"))
            parsed = _parse_model_response(item.get("response", ""))
            meta = prepared_meta.get(item_id, {})
            source_item = source_map.get(item_id, {})
            final_answer = parsed["final_answer"] or str(meta.get("source_answer", "")).strip()
            rewritten_text = _compose_assistant_response(parsed["cot"], final_answer)

            output = dict(source_item) if isinstance(source_item, dict) else {"id": item_id}
            output["id"] = item_id
            output["conversations"] = _replace_last_assistant_message(output.get("conversations"), rewritten_text)
            output["dsv32_solution_content"] = final_answer
            output["dsv32_reasoning_content"] = parsed["cot"]
            output["raw_response"] = item.get("response", "")
            fout.write(json.dumps(output, ensure_ascii=False) + "\n")
            written += 1
    return written


async def run_offline_inference(
    repo_root: Path,
    shard_prepared_jsonl: Path,
    shard_model_output_jsonl: Path,
    model_path: str,
    tp_size: int,
    dp_size: int,
    max_inflight: int,
    enable_dp_attention: bool,
    temperature: float,
    top_p: float,
    max_new_tokens: int,
    resume: bool,
):
    BatchInferenceEngine = _load_batch_inference_engine(repo_root)
    sampling_params = {
        "temperature": temperature,
        "top_p": top_p,
        "max_new_tokens": max_new_tokens,
    }
    async with BatchInferenceEngine(
        model_path=model_path,
        tp_size=tp_size,
        dp_size=dp_size,
        max_inflight=max_inflight,
        enable_dp_attention=enable_dp_attention,
    ) as engine:
        await engine.run(
            input_file=str(shard_prepared_jsonl),
            output_file=str(shard_model_output_jsonl),
            sampling_params=sampling_params,
            resume=resume,
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Offline sharded inference for thinking-to-cot conversion.")
    parser.add_argument("--source-jsonl", required=True, help="Normalized source JSONL with prompt and _meta.")
    parser.add_argument("--prepared-jsonl", required=True, help="Prepared JSONL from preprocess stage.")
    parser.add_argument("--shard-prepared-jsonl", required=True, help="Shard-local prepared JSONL path.")
    parser.add_argument("--shard-model-output-jsonl", required=True, help="Shard-local model output JSONL path.")
    parser.add_argument("--shard-final-jsonl", required=True, help="Shard-local final merged JSONL path.")
    parser.add_argument("--num-shards", type=int, required=True, help="Total number of shards/nodes.")
    parser.add_argument("--shard-rank", type=int, required=True, help="Current shard rank, [0, num_shards).")
    parser.add_argument("--model-path", required=True, help="Local model path for SGLang.")
    parser.add_argument("--tp-size", type=int, default=1, help="Tensor parallel size.")
    parser.add_argument("--dp-size", type=int, default=1, help="Data parallel size.")
    parser.add_argument("--max-inflight", type=int, default=512, help="Offline worker concurrency.")
    parser.add_argument("--enable-dp-attention", action="store_true", help="Enable SGLang DP Attention.")
    parser.add_argument("--temperature", type=float, default=0.2, help="Sampling temperature.")
    parser.add_argument("--top-p", type=float, default=0.95, help="Sampling top-p.")
    parser.add_argument("--max-new-tokens", type=int, default=8192, help="Maximum new tokens.")
    parser.add_argument("--resume", action="store_true", help="Resume using existing shard model output.")
    parser.add_argument("--skip-build-shard-prepared", action="store_true", help="Reuse existing shard prepared file.")
    parser.add_argument("--merge-only", action="store_true", help="Only run merge stage.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.num_shards <= 0:
        raise ValueError("--num-shards must be > 0")
    if args.shard_rank < 0 or args.shard_rank >= args.num_shards:
        raise ValueError("--shard-rank must be in [0, --num-shards)")

    repo_root = Path(__file__).resolve().parents[2]
    source_jsonl = Path(args.source_jsonl)
    prepared_jsonl = Path(args.prepared_jsonl)
    shard_prepared_jsonl = Path(args.shard_prepared_jsonl)
    shard_model_output_jsonl = Path(args.shard_model_output_jsonl)
    shard_final_jsonl = Path(args.shard_final_jsonl)

    if args.skip_build_shard_prepared:
        if not shard_prepared_jsonl.exists():
            raise FileNotFoundError(
                f"--skip-build-shard-prepared is set, but missing file: {shard_prepared_jsonl}"
            )
        print(f"[shard] skipped build, reuse={shard_prepared_jsonl}")
    else:
        total, kept = build_shard_prepared(
            prepared_jsonl=prepared_jsonl,
            shard_prepared_jsonl=shard_prepared_jsonl,
            shard_rank=args.shard_rank,
            num_shards=args.num_shards,
        )
        print(
            f"[shard] rank={args.shard_rank}/{args.num_shards}, total={total}, kept={kept}, "
            f"output={shard_prepared_jsonl}"
        )
        if kept == 0:
            print("[run] no samples for this shard, skip inference.")
            return

    if not args.merge_only:
        asyncio.run(
            run_offline_inference(
                repo_root=repo_root,
                shard_prepared_jsonl=shard_prepared_jsonl,
                shard_model_output_jsonl=shard_model_output_jsonl,
                model_path=args.model_path,
                tp_size=args.tp_size,
                dp_size=args.dp_size,
                max_inflight=args.max_inflight,
                enable_dp_attention=args.enable_dp_attention,
                temperature=args.temperature,
                top_p=args.top_p,
                max_new_tokens=args.max_new_tokens,
                resume=args.resume,
            )
        )

    merged = merge_shard_output(
        source_jsonl=source_jsonl,
        shard_prepared_jsonl=shard_prepared_jsonl,
        shard_model_output_jsonl=shard_model_output_jsonl,
        shard_final_jsonl=shard_final_jsonl,
    )
    print(f"[merge] rank={args.shard_rank}/{args.num_shards}, written={merged}, output={shard_final_jsonl}")


if __name__ == "__main__":
    main()
