import argparse
import json
import multiprocessing as mp
import re
import sys
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

try:
    import orjson  # type: ignore
except ImportError:
    orjson = None


BEGIN_THOUGHT = "<|begin_of_thought|>"
END_THOUGHT = "<|end_of_thought|>"
THOUGHT_PATTERN = re.compile(
    re.escape(BEGIN_THOUGHT) + r"\s*(.*?)\s*" + re.escape(END_THOUGHT),
    flags=re.DOTALL,
)


def _safe_json_loads(line: str) -> Optional[Dict[str, Any]]:
    try:
        if orjson is not None:
            data = orjson.loads(line)
        else:
            data = json.loads(line)
        if isinstance(data, dict):
            return data
    except Exception:
        return None
    return None


def _extract_last_human_message(conversations: List[Dict[str, Any]]) -> str:
    for message in reversed(conversations):
        role = str(message.get("from", "")).strip().lower()
        if role in {"human", "user"}:
            return str(message.get("value", "")).strip()
    return ""


def _extract_last_assistant_message(conversations: List[Dict[str, Any]]) -> str:
    for message in reversed(conversations):
        role = str(message.get("from", "")).strip().lower()
        if role in {"gpt", "assistant"}:
            return str(message.get("value", "")).strip()
    return ""


def _extract_thinking_and_answer(assistant_text: str) -> Tuple[str, str]:
    if not assistant_text:
        return "", ""

    match = THOUGHT_PATTERN.search(assistant_text)
    if not match:
        # Fallback for plain single-turn dialogs without thought tags.
        plain_text = assistant_text.strip()
        return plain_text, plain_text

    thinking = match.group(1).strip()
    final_answer = assistant_text[match.end() :].strip()
    return thinking, final_answer


def _iter_nonempty_lines(raw_jsonl: Path) -> Iterator[Tuple[int, str]]:
    with raw_jsonl.open("r", encoding="utf-8", buffering=1024 * 1024) as fin:
        for line_idx, line in enumerate(fin):
            if not line or line == "\n":
                continue
            yield line_idx, line


def _process_line_for_normalize(task: Tuple[int, str]) -> Optional[Dict[str, Any]]:
    line_idx, line = task
    sample = _safe_json_loads(line)
    if sample is None:
        return None

    conversations = sample.get("conversations")
    if not isinstance(conversations, list) or not conversations:
        return None

    question = _extract_last_human_message(conversations)
    assistant_text = _extract_last_assistant_message(conversations)
    thinking, final_answer = _extract_thinking_and_answer(assistant_text)
    if not thinking:
        return None

    sample_id = sample.get("id", sample.get("index", f"line-{line_idx}"))
    sample["id"] = str(sample_id)
    sample["question"] = question
    sample["thinking"] = thinking
    sample["final_answer"] = final_answer
    return sample


def _write_jsonl_record(fout, record: Dict[str, Any]) -> None:
    if orjson is not None:
        fout.write(orjson.dumps(record, option=orjson.OPT_APPEND_NEWLINE).decode("utf-8"))
    else:
        fout.write(json.dumps(record, ensure_ascii=False) + "\n")


def normalize_raw_jsonl(
    raw_jsonl: Path,
    normalized_jsonl: Path,
    workers: int = 1,
    chunksize: int = 256,
) -> Tuple[int, int]:
    normalized_jsonl.parent.mkdir(parents=True, exist_ok=True)
    total = 0
    written = 0

    with normalized_jsonl.open("w", encoding="utf-8", buffering=1024 * 1024) as fout:
        if workers <= 1:
            for task in _iter_nonempty_lines(raw_jsonl):
                total += 1
                sample = _process_line_for_normalize(task)
                if sample is None:
                    continue
                _write_jsonl_record(fout, sample)
                written += 1
            return total, written

        start_method = "fork" if sys.platform != "win32" else "spawn"
        context = mp.get_context(start_method)
        with context.Pool(processes=workers) as pool:
            for sample in pool.imap(_process_line_for_normalize, _iter_nonempty_lines(raw_jsonl), chunksize=chunksize):
                total += 1
                if sample is None:
                    continue
                _write_jsonl_record(fout, sample)
                written += 1

    return total, written


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Normalize raw conversations JSONL into thinking-conversion input format."
    )
    parser.add_argument("--raw-jsonl", required=True, help="Raw input JSONL containing `conversations`.")
    parser.add_argument(
        "--normalized-jsonl",
        required=True,
        help="Output JSONL path with `id/question/thinking/final_answer` fields.",
    )
    parser.add_argument("--workers", type=int, default=1, help="Number of preprocess workers. Use >1 for multiprocessing.")
    parser.add_argument("--chunksize", type=int, default=256, help="Multiprocessing chunksize for line batches.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    raw_jsonl = Path(args.raw_jsonl)
    normalized_jsonl = Path(args.normalized_jsonl)
    total, written = normalize_raw_jsonl(
        raw_jsonl=raw_jsonl,
        normalized_jsonl=normalized_jsonl,
        workers=max(1, args.workers),
        chunksize=max(1, args.chunksize),
    )
    print(
        f"[normalize] total={total}, written={written}, workers={max(1, args.workers)}, "
        f"chunksize={max(1, args.chunksize)}, output={normalized_jsonl}"
    )


if __name__ == "__main__":
    main()
