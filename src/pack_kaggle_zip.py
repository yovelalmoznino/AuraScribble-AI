from __future__ import annotations

import argparse
import zipfile
from pathlib import Path

_EXCLUDE_DIRS = {".git", ".venv", "__pycache__", "output", "models", "data/processed"}
_EXCLUDE_FILES = {".pt", ".pyc", ".onnx"}


def _should_include(path: Path, root: Path) -> bool:
    rel = path.relative_to(root)
    for part in rel.parts:
        if part in _EXCLUDE_DIRS:
            return False
    if path.suffix.lower() in _EXCLUDE_FILES:
        return False
    if path.name == "checkpoint_best.pt":
        return False
    return True


def pack_training_zip(source: Path, output: Path) -> None:
    source = source.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output, "w", zipfile.ZIP_DEFLATED) as zf:
        for path in sorted(source.rglob("*")):
            if not path.is_file():
                continue
            if not _should_include(path, source):
                continue
            arcname = Path("handwriting-model") / path.relative_to(source)
            zf.write(path, arcname.as_posix())
    print(f"Packed {output} ({output.stat().st_size // 1024} KB)")


def main() -> None:
    parser = argparse.ArgumentParser(description="Pack handwriting-model folder for Kaggle upload")
    parser.add_argument("--source", type=Path, default=Path("."))
    parser.add_argument("--output", type=Path, default=Path("handwriting_training_bundle.zip"))
    args = parser.parse_args()
    pack_training_zip(args.source, args.output)


if __name__ == "__main__":
    main()
