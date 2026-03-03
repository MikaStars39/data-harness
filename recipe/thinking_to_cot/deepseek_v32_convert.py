import argparse
import asyncio
import importlib.util
import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Optional, Tuple

from transformers import AutoTokenizer


def _load_online_inference_engine(repo_root: Path):
    package_name = "data_harness_runtime"
    package_dir = repo_root / "data-harness"
    online_path = package_dir / "online.py"

    if not online_path.exists():
        raise FileNotFoundError("Cannot find `data-harness/online.py`.")

    if package_name not in sys.modules:
        package_spec = importlib.util.spec_from_loader(package_name, loader=None)
        package_module = importlib.util.module_from_spec(package_spec)
        package_module.__path__ = [str(package_dir)]
        sys.modules[package_name] = package_module

    online_spec = importlib.util.spec_from_file_location(f"{package_name}.online", online_path)
    online_module = importlib.util.module_from_spec(online_spec)
    sys.modules[f"{package_name}.online"] = online_module
    assert online_spec.loader is not None
    online_spec.loader.exec_module(online_module)

    return online_module.OnlineBatchInferenceEngine, online_module.APIConfig

def _load_batch_inference_engine(repo_root: Path):
    """
    Load BatchInferenceEngine from `data-harness/offline.py`.

    The folder name contains a hyphen, so we dynamically create a virtual package
    and import `base.py` and `offline.py` under that package namespace.
    """
    package_name = "data_harness_runtime"
    package_dir = repo_root / "data-harness"
    base_path = package_dir / "base.py"
    offline_path = package_dir / "offline.py"

    if not base_path.exists() or not offline_path.exists():
        raise FileNotFoundError("Cannot find `data-harness/base.py` or `data-harness/offline.py`.")

    package_spec = importlib.util.spec_from_loader(package_name, loader=None)
    package_module = importlib.util.module_from_spec(package_spec)
    package_module.__path__ = [str(package_dir)]  # Needed for relative imports.
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


THINKING_KEYS = [
    "thinking",
    "reasoning",
    "reasoning_content",
    "thought",
    "analysis",
    "chain_of_thought",
]

QUESTION_KEYS = [
    "question",
    "prompt",
    "query",
    "instruction",
    "input",
]

ANSWER_KEYS = [
    "answer",
    "response",
    "output",
    "final_answer",
]


def _first_nonempty(data: Dict[str, Any], keys: Iterable[str]) -> str:
    for key in keys:
        value = data.get(key)
        if value is None:
            continue
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def _build_convert_prompt(
    question: str,
    thinking: str,
    answer: str,
) -> str:
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


def _make_chat_formatter(model_path: str) -> Callable[[str], str]:
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    def _format(prompt: str) -> str:
        messages = [
            {"role": "system", "content": "You transform thinking traces into clean chain-of-thought data."},
            {"role": "user", "content": prompt},
        ]
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

    return _format


def prepare_input_jsonl(
    source_jsonl: Path,
    prepared_jsonl: Path,
    prompt_formatter: Optional[Callable[[str], str]] = None,
) -> Tuple[int, int]:
    prepared_jsonl.parent.mkdir(parents=True, exist_ok=True)
    total, valid = 0, 0

    with source_jsonl.open("r", encoding="utf-8") as fin, prepared_jsonl.open("w", encoding="utf-8") as fout:
        for idx, line in enumerate(fin):
            if not line.strip():
                continue
            total += 1
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue

            item_id = str(item.get("id", f"line-{idx}"))
            raw_prompt = item.get("prompt")
            meta = item.get("_meta") if isinstance(item.get("_meta"), dict) else {}

            # Fast path: preprocessing already prepared prompt and meta.
            if not isinstance(raw_prompt, str) or not raw_prompt.strip():
                thinking = _first_nonempty(item, THINKING_KEYS)
                if not thinking:
                    continue
                question = _first_nonempty(item, QUESTION_KEYS)
                answer = _first_nonempty(item, ANSWER_KEYS)
                raw_prompt = _build_convert_prompt(
                    question=question,
                    thinking=thinking,
                    answer=answer,
                )
                meta = {
                    "question": question,
                    "source_answer": answer,
                    "source_thinking_chars": len(thinking),
                }

            prompt = prompt_formatter(raw_prompt) if prompt_formatter else raw_prompt

            payload = {
                "id": item_id,
                "prompt": prompt,
                "_meta": meta,
            }
            fout.write(json.dumps(payload, ensure_ascii=False) + "\n")
            valid += 1

    return total, valid


def _parse_model_response(raw_text: str) -> Dict[str, str]:
    raw_text = (raw_text or "").strip()
    if not raw_text:
        return {"cot": "", "final_answer": ""}

    # Prefer strict JSON output; fallback to plain text when model drifts.
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


def _replace_last_assistant_message(
    conversations: Any,
    rewritten_text: str,
) -> list:
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


def merge_output(source_jsonl: Path, prepared_jsonl: Path, model_output_jsonl: Path, final_jsonl: Path) -> int:
    final_jsonl.parent.mkdir(parents=True, exist_ok=True)
    prepared_map: Dict[str, Dict[str, Any]] = {}
    source_map: Dict[str, Dict[str, Any]] = {}
    written = 0

    with source_jsonl.open("r", encoding="utf-8") as fin:
        for idx, line in enumerate(fin):
            if not line.strip():
                continue
            item = json.loads(line)
            item_id = str(item.get("id", f"line-{idx}"))
            source_map[item_id] = item

    with prepared_jsonl.open("r", encoding="utf-8") as fin:
        for line in fin:
            if not line.strip():
                continue
            item = json.loads(line)
            prepared_map[str(item["id"])] = item

    with model_output_jsonl.open("r", encoding="utf-8") as fin, final_jsonl.open("w", encoding="utf-8") as fout:
        for line in fin:
            if not line.strip():
                continue
            item = json.loads(line)
            item_id = str(item.get("id"))
            base_item = prepared_map.get(item_id, {})
            parsed = _parse_model_response(item.get("response", ""))
            meta = base_item.get("_meta", {})
            source_item = source_map.get(item_id, {})

            final_answer = parsed["final_answer"] or meta.get("source_answer", "")
            rewritten_text = _compose_assistant_response(parsed["cot"], final_answer)

            output = dict(source_item) if isinstance(source_item, dict) else {"id": item_id}
            output["id"] = item_id
            output["conversations"] = _replace_last_assistant_message(
                output.get("conversations"),
                rewritten_text,
            )

            # Keep structured fields for audit/debug while preserving final conversation shape.
            output["dsv32_solution_content"] = final_answer
            output["dsv32_reasoning_content"] = parsed["cot"]
            output["raw_response"] = item.get("response", "")
            fout.write(json.dumps(output, ensure_ascii=False) + "\n")
            written += 1

    return written


async def run_inference(
    repo_root: Path,
    prepared_jsonl: Path,
    model_output_jsonl: Path,
    model_path: str,
    tp_size: int,
    dp_size: int,
    max_inflight: int,
    enable_dp_attention: bool,
    temperature: float,
    top_p: float,
    max_new_tokens: int,
    resume: bool,
    engine_type: str,
    api_key: str,
    base_url: str,
):
    sampling_params = {
        "temperature": temperature,
        "top_p": top_p,
        "max_new_tokens": max_new_tokens,
    }

    if engine_type == "online":
        OnlineBatchInferenceEngine, APIConfig = _load_online_inference_engine(repo_root)
        config = APIConfig(
            api_key=api_key,
            base_url=base_url.rstrip('/'),
            model=model_path
        )
        engine = OnlineBatchInferenceEngine(config, concurrency=max_inflight)
        # Note: online engine uses "max_tokens" instead of "max_new_tokens"
        sampling_params["max_tokens"] = sampling_params.pop("max_new_tokens")
        await engine.run(
            input_file=str(prepared_jsonl),
            output_file=str(model_output_jsonl),
            sampling_params=sampling_params,
        )
    else:
        BatchInferenceEngine = _load_batch_inference_engine(repo_root)
        async with BatchInferenceEngine(
            model_path=model_path,
            tp_size=tp_size,
            dp_size=dp_size,
            max_inflight=max_inflight,
            enable_dp_attention=enable_dp_attention,
        ) as engine:
            await engine.run(
                input_file=str(prepared_jsonl),
                output_file=str(model_output_jsonl),
                sampling_params=sampling_params,
                resume=resume,
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert thinking traces into non-thinking CoT using DeepSeek v3.2 offline pipeline."
    )
    parser.add_argument("--source-jsonl", required=True, help="Input JSONL containing thinking traces.")
    parser.add_argument("--prepared-jsonl", required=True, help="Prepared prompt JSONL for batch inference.")
    parser.add_argument("--model-output-jsonl", required=True, help="Raw model output JSONL path.")
    parser.add_argument("--final-jsonl", required=True, help="Final merged output JSONL path.")
    parser.add_argument("--model-path", default="deepseek-ai/DeepSeek-V3.2", help="Model path for SGLang.")
    parser.add_argument("--tp-size", type=int, default=1, help="Tensor parallel size.")
    parser.add_argument("--dp-size", type=int, default=1, help="Data parallel size.")
    parser.add_argument("--max-inflight", type=int, default=256, help="Offline worker concurrency.")
    parser.add_argument("--enable-dp-attention", action="store_true", help="Enable SGLang DP Attention for higher throughput.")
    parser.add_argument("--temperature", type=float, default=0.2, help="Sampling temperature.")
    parser.add_argument("--top-p", type=float, default=0.95, help="Sampling top-p.")
    parser.add_argument("--max-new-tokens", type=int, default=2048, help="Maximum new tokens.")
    parser.add_argument(
        "--apply-chat-template",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Whether to format prompt with tokenizer chat template.",
    )
    parser.add_argument(
        "--chat-template-model-path",
        default="",
        help="Tokenizer path used for chat template. Defaults to --model-path.",
    )
    parser.add_argument("--resume", action="store_true", help="Resume from existing model output file.")
    parser.add_argument("--prepare-only", action="store_true", help="Only build prepared JSONL and exit.")
    
    # Online Engine args
    parser.add_argument("--engine-type", type=str, default="offline", choices=["offline", "online"], help="Inference engine to use.")
    parser.add_argument("--api-key", type=str, default="", help="API key for online engine.")
    parser.add_argument("--base-url", type=str, default="", help="Base URL for online engine.")

    return parser.parse_args()


def main():
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[2]

    source_jsonl = Path(args.source_jsonl)
    prepared_jsonl = Path(args.prepared_jsonl)
    model_output_jsonl = Path(args.model_output_jsonl)
    final_jsonl = Path(args.final_jsonl)

    prompt_formatter: Optional[Callable[[str], str]] = None
    if args.apply_chat_template:
        template_model_path = args.chat_template_model_path or args.model_path
        try:
            prompt_formatter = _make_chat_formatter(template_model_path)
            print(f"[prepare] chat template enabled with tokenizer={template_model_path}")
        except Exception as exc:
            print(f"[prepare] chat template unavailable ({exc}), fallback to raw prompt.")

    total, valid = prepare_input_jsonl(source_jsonl, prepared_jsonl, prompt_formatter=prompt_formatter)
    print(f"[prepare] total={total}, valid_with_thinking={valid}, output={prepared_jsonl}")

    if args.prepare_only:
        return
    if valid == 0:
        print("[run] no valid items to process, skip inference.")
        return

    asyncio.run(
        run_inference(
            repo_root=repo_root,
            prepared_jsonl=prepared_jsonl,
            model_output_jsonl=model_output_jsonl,
            model_path=args.model_path,
            tp_size=args.tp_size,
            dp_size=args.dp_size,
            max_inflight=args.max_inflight,
            enable_dp_attention=args.enable_dp_attention,
            temperature=args.temperature,
            top_p=args.top_p,
            max_new_tokens=args.max_new_tokens,
            resume=args.resume,
            engine_type=args.engine_type,
            api_key=args.api_key,
            base_url=args.base_url,
        )
    )
    merged = merge_output(source_jsonl, prepared_jsonl, model_output_jsonl, final_jsonl)
    print(f"[merge] written={merged}, output={final_jsonl}")


if __name__ == "__main__":
    main()
