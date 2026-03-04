import argparse
import asyncio
import importlib.util
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Tuple


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


def _build_convert_prompt(question: str, thinking: str, answer: str) -> str:
    answer_block = answer if answer else "(No final answer provided in source.)"
    question_block = question if question else "(No user question provided in source.)"
    source_chars = len(thinking)
    min_chars = max(200, int(source_chars * 0.90))
    return (
        "You are an expert data rewriter.\n"
        "You convert a `thinking model trace` into explicit chain-of-thought for non-thinking model data.\n\n"
        "Requirements:\n"
        "1) Language consistency is mandatory: output `cot` following the source response language.\n"
        "2) Preserve details and do not over-compress. Keep all key steps, derivations, checks, and transitions.\n"
        f"3) Length constraint: `cot` must be at least {min_chars} characters (source thinking length={source_chars}).\n"
        "4) Prefer explicit step-by-step structure (e.g., Step 1/Step 2) and keep equations/calculations complete.\n"
        "5) Remove only obvious model self-talk or duplicate loops. Do not drop useful reasoning content.\n"
        "6) Ensure consistency with the final answer when provided. Both `cot` and `final_answer` must be completely populated without truncation.\n"
        "7) Output ONLY valid JSON.\n\n"
        "Output schema:\n"
        "{\n"
        "  \"cot\": \"<long explicit step-by-step chain-of-thought, e.g., Step 1, Step 2...>\",\n"
        "  \"final_answer\": \"<complete final answer, e.g., boxed result>\"\n"
        "}\n\n"
        f"Question:\n{question_block}\n\n"
        f"Original thinking trace:\n{thinking}\n\n"
        f"Reference final answer:\n{answer_block}\n"
    )


def prepare_shard_input(shard_source_jsonl: Path, shard_prepared_jsonl: Path) -> Tuple[int, int]:
    shard_prepared_jsonl.parent.mkdir(parents=True, exist_ok=True)
    total = 0
    valid = 0
    with shard_source_jsonl.open("r", encoding="utf-8") as fin, shard_prepared_jsonl.open("w", encoding="utf-8") as fout:
        for idx, line in enumerate(fin):
            if not line.strip():
                continue
            total += 1
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue

            item_id = str(item.get("id", f"line-{idx}"))
            prompt = item.get("prompt")
            meta = item.get("_meta") if isinstance(item.get("_meta"), dict) else {}

            if not isinstance(prompt, str) or not prompt.strip():
                thinking = str(item.get("thinking", "")).strip()
                if not thinking:
                    continue
                question = str(item.get("question", "")).strip()
                answer = str(item.get("final_answer", "")).strip()
                prompt = _build_convert_prompt(question=question, thinking=thinking, answer=answer)
                meta = {
                    "question": question,
                    "source_answer": answer,
                    "source_thinking_chars": len(thinking),
                }

            fout.write(json.dumps({"id": item_id, "prompt": prompt, "_meta": meta}, ensure_ascii=False) + "\n")
            valid += 1
    return total, valid


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
            return {
                "cot": str(obj.get("cot", "")).strip(),
                "final_answer": str(obj.get("final_answer", "")).strip(),
            }
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


def merge_single_shard_output(
    shard_source_jsonl: Path,
    shard_prepared_jsonl: Path,
    shard_model_output_jsonl: Path,
    shard_final_jsonl: Path,
) -> int:
    shard_final_jsonl.parent.mkdir(parents=True, exist_ok=True)
    source_map: Dict[str, Dict[str, Any]] = {}
    prepared_meta: Dict[str, Dict[str, Any]] = {}
    written = 0

    with shard_source_jsonl.open("r", encoding="utf-8") as fin:
        for idx, line in enumerate(fin):
            if not line.strip():
                continue
            item = json.loads(line)
            source_map[str(item.get("id", f"line-{idx}"))] = item

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
            source_item = source_map.get(item_id, {})
            meta = prepared_meta.get(item_id, {})
            parsed = _parse_model_response(item.get("response", ""))
            final_answer = parsed["final_answer"] or str(meta.get("source_answer", "")).strip()
            rewritten = _compose_assistant_response(parsed["cot"], final_answer)

            output = dict(source_item) if isinstance(source_item, dict) else {"id": item_id}
            output["id"] = item_id
            output["conversations"] = _replace_last_assistant_message(output.get("conversations"), rewritten)
            output["dsv32_solution_content"] = final_answer
            output["dsv32_reasoning_content"] = parsed["cot"]
            output["raw_response"] = item.get("response", "")
            fout.write(json.dumps(output, ensure_ascii=False) + "\n")
            written += 1
    return written


async def run_offline(
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
) -> None:
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
    parser = argparse.ArgumentParser(description="Run offline inference on a single shard JSONL.")
    parser.add_argument("--shard-source-jsonl", required=True, help="Input shard JSONL for this node.")
    parser.add_argument("--shard-prepared-jsonl", required=True, help="Prepared shard JSONL path.")
    parser.add_argument("--shard-model-output-jsonl", required=True, help="Shard model output path.")
    parser.add_argument("--shard-final-jsonl", required=True, help="Shard final merged output path.")
    parser.add_argument("--model-path", required=True, help="Local model path for SGLang.")
    parser.add_argument("--tp-size", type=int, default=1, help="Tensor parallel size.")
    parser.add_argument("--dp-size", type=int, default=1, help="Data parallel size.")
    parser.add_argument("--max-inflight", type=int, default=512, help="Offline worker concurrency.")
    parser.add_argument("--enable-dp-attention", action="store_true", help="Enable DP Attention.")
    parser.add_argument("--temperature", type=float, default=0.2, help="Sampling temperature.")
    parser.add_argument("--top-p", type=float, default=0.95, help="Sampling top-p.")
    parser.add_argument("--max-new-tokens", type=int, default=8192, help="Max generation tokens.")
    parser.add_argument("--resume", action="store_true", help="Resume using existing model output.")
    parser.add_argument("--skip-prepare", action="store_true", help="Reuse shard prepared file directly.")
    parser.add_argument("--merge-only", action="store_true", help="Only run merge stage.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[2]
    shard_source_jsonl = Path(args.shard_source_jsonl)
    shard_prepared_jsonl = Path(args.shard_prepared_jsonl)
    shard_model_output_jsonl = Path(args.shard_model_output_jsonl)
    shard_final_jsonl = Path(args.shard_final_jsonl)

    if args.skip_prepare:
        if not shard_prepared_jsonl.exists():
            raise FileNotFoundError(
                f"--skip-prepare is set, but prepared file does not exist: {shard_prepared_jsonl}"
            )
        print(f"[prepare] skipped, reuse={shard_prepared_jsonl}")
    else:
        total, valid = prepare_shard_input(shard_source_jsonl, shard_prepared_jsonl)
        print(f"[prepare] total={total}, valid={valid}, output={shard_prepared_jsonl}")
        if valid == 0:
            print("[run] no valid samples in shard, skip.")
            return

    if not args.merge_only:
        asyncio.run(
            run_offline(
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

    merged = merge_single_shard_output(
        shard_source_jsonl=shard_source_jsonl,
        shard_prepared_jsonl=shard_prepared_jsonl,
        shard_model_output_jsonl=shard_model_output_jsonl,
        shard_final_jsonl=shard_final_jsonl,
    )
    print(f"[merge] written={merged}, output={shard_final_jsonl}")


if __name__ == "__main__":
    main()
