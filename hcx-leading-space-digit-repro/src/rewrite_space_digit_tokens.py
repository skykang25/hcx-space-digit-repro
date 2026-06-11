#!/usr/bin/env python3
"""Utilities for decode-equivalent leading-space digit token rewrites.

This script is intentionally small and model-specific. It documents the token
ID rewrite used in the HyperCLOVAX-SEED-Think-14B GSM8K reproduction note.

The visible decoded text is intended to remain unchanged:

    " 2" -> [109896]    # original leading-space digit single token
    " 2" -> [220, 17]  # rewritten space token + digit token
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


SPACE_DIGIT_REWRITE_MAP: dict[int, list[int]] = {
    109658: [220, 15],  # " 0"
    103647: [220, 16],  # " 1"
    109896: [220, 17],  # " 2"
    103590: [220, 18],  # " 3"
    101709: [220, 19],  # " 4"
    110217: [220, 20],  # " 5"
    105778: [220, 21],  # " 6"
    103650: [220, 22],  # " 7"
    101994: [220, 23],  # " 8"
    102409: [220, 24],  # " 9"
}


def rewrite_space_digit_token_ids(token_ids: list[int]) -> tuple[list[int], list[dict[str, Any]]]:
    """Rewrite leading-space digit single tokens.

    Returns the rewritten token IDs plus a list of rewrite events with original
    positions. Non-matching tokens are copied unchanged.
    """
    rewritten: list[int] = []
    events: list[dict[str, Any]] = []

    for index, token_id in enumerate(token_ids):
        replacement = SPACE_DIGIT_REWRITE_MAP.get(token_id)
        if replacement is None:
            rewritten.append(token_id)
            continue

        rewritten.extend(replacement)
        events.append(
            {
                "index": index,
                "original_token_id": token_id,
                "rewritten_token_ids": replacement,
            }
        )

    return rewritten, events


def parse_ids(raw: str) -> list[int]:
    return [int(part.strip()) for part in raw.split(",") if part.strip()]


def load_ids_json(path: Path) -> list[int]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, list):
        return [int(item) for item in data]
    if isinstance(data, dict):
        for key in ("prompt_token_ids", "token_ids", "input_ids"):
            value = data.get(key)
            if isinstance(value, list):
                return [int(item) for item in value]
    raise ValueError("JSON must be a list of token IDs or contain prompt_token_ids/token_ids/input_ids.")


def load_tokenizer(name: str):
    try:
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise SystemExit("transformers is required for --text or --check-map.") from exc
    return AutoTokenizer.from_pretrained(name)


def check_map_decode_equivalence(tokenizer_name: str) -> list[dict[str, Any]]:
    tokenizer = load_tokenizer(tokenizer_name)
    checks = []
    for original_id, replacement in sorted(SPACE_DIGIT_REWRITE_MAP.items()):
        original_text = tokenizer.decode([original_id])
        rewritten_text = tokenizer.decode(replacement)
        checks.append(
            {
                "original_token_id": original_id,
                "original_text": original_text,
                "rewritten_token_ids": replacement,
                "rewritten_text": rewritten_text,
                "decode_equivalent": original_text == rewritten_text,
            }
        )
    return checks


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--ids", help="Comma-separated token IDs, e.g. '109896,101709'.")
    source.add_argument("--ids-json", type=Path, help="JSON list or object containing prompt_token_ids.")
    source.add_argument("--text", help="Text to tokenize before applying the rewrite.")
    parser.add_argument("--tokenizer", default="naver-hyperclovax/HyperCLOVAX-SEED-Think-14B")
    parser.add_argument("--check-map", action="store_true", help="Check tokenizer decode equivalence for the map.")
    args = parser.parse_args()

    output: dict[str, Any] = {
        "rewrite_map": SPACE_DIGIT_REWRITE_MAP,
    }

    if args.check_map:
        output["decode_equivalence_checks"] = check_map_decode_equivalence(args.tokenizer)

    if args.text is not None:
        tokenizer = load_tokenizer(args.tokenizer)
        token_ids = tokenizer.encode(args.text, add_special_tokens=False)
        rewritten, events = rewrite_space_digit_token_ids(token_ids)
        output.update(
            {
                "input_text": args.text,
                "original_token_ids": token_ids,
                "rewritten_token_ids": rewritten,
                "rewrite_events": events,
                "original_decoded": tokenizer.decode(token_ids),
                "rewritten_decoded": tokenizer.decode(rewritten),
                "decode_equivalent": tokenizer.decode(token_ids) == tokenizer.decode(rewritten),
            }
        )
    elif args.ids is not None:
        token_ids = parse_ids(args.ids)
        rewritten, events = rewrite_space_digit_token_ids(token_ids)
        output.update(
            {
                "original_token_ids": token_ids,
                "rewritten_token_ids": rewritten,
                "rewrite_events": events,
            }
        )
    elif args.ids_json is not None:
        token_ids = load_ids_json(args.ids_json)
        rewritten, events = rewrite_space_digit_token_ids(token_ids)
        output.update(
            {
                "source_json": str(args.ids_json),
                "original_token_ids": token_ids,
                "rewritten_token_ids": rewritten,
                "rewrite_events": events,
            }
        )

    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

