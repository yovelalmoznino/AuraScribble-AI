"""Trigger the AuraScribble Kaggle kernel and wait for completion.

Reads KAGGLE_USERNAME + KAGGLE_KEY from env (the kaggle CLI auto-detects them).
Pushes the latest notebook from kaggle/KAGGLE_TRAIN_V8.ipynb to the user's kernel,
then polls until status is complete.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path


def run(cmd: list[str], **kwargs) -> subprocess.CompletedProcess:
    print("$", " ".join(cmd))
    return subprocess.run(cmd, check=True, capture_output=True, text=True, **kwargs)


def write_metadata(slug: str, title: str, notebook_path: Path, dataset_slug: str) -> Path:
    """Build the kernel-metadata.json that `kaggle kernels push` expects."""
    meta = {
        "id": slug,
        "title": title,
        "code_file": notebook_path.name,
        "language": "python",
        "kernel_type": "notebook",
        "is_private": True,
        "enable_gpu": True,
        "enable_tpu": False,
        "enable_internet": True,
        "dataset_sources": [dataset_slug],
        "competition_sources": [],
        "kernel_sources": [],
    }
    out = notebook_path.parent / "kernel-metadata.json"
    out.write_text(json.dumps(meta, indent=2))
    return out


def push_kernel(notebook_path: Path) -> None:
    folder = notebook_path.parent
    run(["kaggle", "kernels", "push", "-p", str(folder)])


def wait_for_completion(slug: str, timeout_min: int = 120, poll_sec: int = 60) -> dict:
    """Poll `kaggle kernels status` until done or timeout."""
    deadline = time.time() + timeout_min * 60
    last_status = None
    while time.time() < deadline:
        try:
            cp = run(["kaggle", "kernels", "status", slug])
        except subprocess.CalledProcessError as e:
            print(f"status check failed (transient): {e.stderr}", file=sys.stderr)
            time.sleep(poll_sec)
            continue

        out = cp.stdout.strip()
        if out != last_status:
            print(out)
            last_status = out

        low = out.lower()
        if "complete" in low and "running" not in low and "queued" not in low:
            return {"status": "complete", "raw": out}
        if "error" in low or "fail" in low or "cancel" in low:
            return {"status": "error", "raw": out}
        time.sleep(poll_sec)

    return {"status": "timeout", "raw": last_status or ""}


def download_outputs(slug: str, dest: Path) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    run(["kaggle", "kernels", "output", slug, "-p", str(dest)])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--notebook",
        type=Path,
        default=Path("tools/handwriting-model/kaggle/KAGGLE_TRAIN_V8.ipynb"),
    )
    ap.add_argument(
        "--slug",
        default=os.environ.get("KAGGLE_KERNEL_SLUG"),
        help="username/kernel-slug (e.g. 'yalmoznino/aurascribble-train')",
    )
    ap.add_argument(
        "--dataset",
        default=os.environ.get("KAGGLE_DATASET_SLUG"),
        help="username/dataset-slug for the bundle (e.g. 'yalmoznino/aurascribble-bundle')",
    )
    ap.add_argument("--title", default="AuraScribble Auto-Retrain V8")
    ap.add_argument("--timeout-min", type=int, default=120)
    ap.add_argument("--output-dir", type=Path, default=Path("kaggle_outputs"))
    args = ap.parse_args()

    if not args.slug or not args.dataset:
        print("ERROR: --slug and --dataset are required (or set KAGGLE_KERNEL_SLUG / KAGGLE_DATASET_SLUG)", file=sys.stderr)
        sys.exit(2)
    if not args.notebook.exists():
        print(f"ERROR: notebook not found: {args.notebook}", file=sys.stderr)
        sys.exit(2)

    print(f"Pushing notebook for kernel {args.slug}")
    write_metadata(args.slug, args.title, args.notebook, args.dataset)
    push_kernel(args.notebook)

    print(f"Waiting up to {args.timeout_min} min for kernel run to finish...")
    result = wait_for_completion(args.slug, timeout_min=args.timeout_min)
    print(f"Final status: {result['status']}")

    if result["status"] != "complete":
        print(f"Kernel did not complete successfully (raw: {result['raw']})", file=sys.stderr)
        sys.exit(1)

    print(f"Downloading kernel outputs to {args.output_dir}")
    download_outputs(args.slug, args.output_dir)

    result_file = args.output_dir / "result.json"
    if result_file.exists():
        print("=== Run result ===")
        print(result_file.read_text())
    else:
        print("WARNING: no result.json in kernel outputs — assume failure")
        sys.exit(1)


if __name__ == "__main__":
    main()
