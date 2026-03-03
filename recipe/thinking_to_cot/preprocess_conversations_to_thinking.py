import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


BEGIN_THOUGHT = "<|begin_of_thought|>"
END_THOUGHT = "<|end_of_thought|>"


def _safe_json_loads(line: str) -> Optional[Dict[str, Any]]:
    try:
        data = json.loads(line)
        if isinstance(data, dict):
            return data
    except json.JSONDecodeError:
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

    pattern = re.compile(
        re.escape(BEGIN_THOUGHT) + r"\s*(.*?)\s*" + re.escape(END_THOUGHT),
        flags=re.DOTALL,
    )
    match = pattern.search(assistant_text)
    if not match:
        # Fallback for plain single-turn dialogs without thought tags.
        plain_text = assistant_text.strip()
        return plain_text, plain_text

    thinking = match.group(1).strip()
    final_answer = assistant_text[match.end() :].strip()
    return thinking, final_answer


def normalize_raw_jsonl(raw_jsonl: Path, normalized_jsonl: Path) -> Tuple[int, int]:
    normalized_jsonl.parent.mkdir(parents=True, exist_ok=True)
    total = 0
    written = 0

    with raw_jsonl.open("r", encoding="utf-8") as fin, normalized_jsonl.open("w", encoding="utf-8") as fout:
        for line_idx, line in enumerate(fin):
            if not line.strip():
                continue

            total += 1
            sample = _safe_json_loads(line)
            if sample is None:
                continue

            conversations = sample.get("conversations")
            if not isinstance(conversations, list) or not conversations:
                continue

            question = _extract_last_human_message(conversations)
            assistant_text = _extract_last_assistant_message(conversations)
            thinking, final_answer = _extract_thinking_and_answer(assistant_text)
            if not thinking:
                continue

            sample_id = sample.get("id", sample.get("index", f"line-{line_idx}"))
            # Keep the original sample structure so the final merge can write back
            # into the original single-turn conversation format.
            output = dict(sample)
            output["id"] = str(sample_id)
            output["question"] = question
            output["thinking"] = thinking
            output["final_answer"] = final_answer
            fout.write(json.dumps(output, ensure_ascii=False) + "\n")
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
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    raw_jsonl = Path(args.raw_jsonl)
    normalized_jsonl = Path(args.normalized_jsonl)
    total, written = normalize_raw_jsonl(raw_jsonl, normalized_jsonl)
    print(f"[normalize] total={total}, written={written}, output={normalized_jsonl}")


if __name__ == "__main__":
    main()
