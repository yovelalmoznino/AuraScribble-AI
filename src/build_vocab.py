from __future__ import annotations

import argparse
from pathlib import Path

from dataset import read_manifest


def _base_vocab_tokens() -> list[str]:
    special = ["<blank>", "<pad>", "<bos>", "<eos>"]
    mode = ["<auto>", "<he>", "<en>", "<math>"]
    digits = [str(d) for d in range(10)]
    latex_basic = list(r"\{}_^+-=()/[]<>|*%&.,;:!? ")
    hebrew = list("אבגדהוזחטיכלמנסעפצקרשת")
    hebrew_final = list("ךםןףץ")
    latin_lower = [chr(c) for c in range(ord("a"), ord("z") + 1)]
    latin_upper = [chr(c) for c in range(ord("A"), ord("Z") + 1)]
    greek_lower = list("αβγδεζηθικλμνξοπρστυφχψω")
    greek_upper = list("ΑΒΓΔΘΛΜΠΣΦΩ")
    math_delim = ["$"]
    return (
        special
        + mode
        + digits
        + latex_basic
        + hebrew
        + hebrew_final
        + latin_lower
        + latin_upper
        + greek_lower
        + greek_upper
        + math_delim
    )


def build_vocab_from_manifest(manifest_path: Path | None) -> list[str]:
    tokens = _base_vocab_tokens()
    seen = set(tokens)
    if manifest_path and manifest_path.exists():
        for sample in read_manifest(manifest_path):
            for ch in sample.text or "":
                if ch not in seen:
                    seen.add(ch)
                    tokens.append(ch)
    return tokens


def write_vocab(path: Path, tokens: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for tok in tokens:
            f.write(f"{tok}\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build handwriting vocab from manifest + base tokens")
    parser.add_argument("--manifest", type=Path, default=Path("data/processed/all.jsonl"))
    parser.add_argument("--output", type=Path, default=Path("configs/vocab.txt"))
    parser.add_argument(
        "--android-assets",
        type=Path,
        default=Path("../../app/src/main/assets/models/handwriting/vocab.txt"),
    )
    args = parser.parse_args()

    tokens = build_vocab_from_manifest(args.manifest if args.manifest.exists() else None)
    write_vocab(args.output, tokens)
    print(f"Wrote {len(tokens)} tokens -> {args.output}")

    android_path = args.android_assets.resolve()
    if android_path.parent.exists():
        write_vocab(android_path, tokens)
        print(f"Copied vocab -> {android_path}")


if __name__ == "__main__":
    main()
