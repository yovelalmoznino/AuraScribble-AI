"""Trigger the AuraScribble Kaggle kernel and wait for completion.

Reads KAGGLE_USERNAME + KAGGLE_KEY from env (the kaggle CLI auto-detects them).
Pushes the latest notebook from kaggle/KAGGLE_TRAIN_V8.ipynb to the user's kernel,
then polls until status is complete.
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import subprocess
import sys
import time
from pathlib import Path


def run(cmd: list[str], **kwargs) -> subprocess.CompletedProcess:
    print("$", " ".join(cmd), flush=True)
    cp = subprocess.run(cmd, capture_output=True, text=True, **kwargs)
    if cp.stdout:
        print(cp.stdout, end="", flush=True)
    if cp.stderr:
        print(cp.stderr, end="", file=sys.stderr, flush=True)
    if cp.returncode != 0:
        raise subprocess.CalledProcessError(
            cp.returncode, cmd, output=cp.stdout, stderr=cp.stderr
        )
    return cp


def write_metadata(slug: str, title: str, notebook_path: Path, dataset_slug: str) -> Path:
    """Build the kernel-metadata.json that `kaggle kernels push` expects.

    Kaggle requires that the title resolves to the same slug as the id, otherwise
    it warns and refuses to push. We derive a slug-safe title from the id's
    kernel-slug component.
    """
    # Derive a slug-safe title from the kernel slug (e.g. "yovelalmoznino/aurascribble-train"
    # → "aurascribble train")
    kernel_slug = slug.split("/", 1)[-1]
    safe_title = kernel_slug.replace("-", " ").strip()
    if not safe_title:
        safe_title = title

    meta = {
        "id": slug,
        "title": safe_title,
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


def inject_firebase_secret(notebook_path: Path) -> None:
    """Replace the __FIREBASE_SA_B64__ placeholder in the notebook with a base64
    encoding of the Firebase service account JSON read from env. Kaggle Secrets
    can't be attached via the API, so we inline the credential into the kernel
    source instead. Kernel is private, only the owner can read it.
    """
    sa_json = os.environ.get("FIREBASE_SERVICE_ACCOUNT_JSON")
    if not sa_json:
        print(
            "WARNING: FIREBASE_SERVICE_ACCOUNT_JSON env var is empty — "
            "kernel will try Kaggle Secrets fallback (which usually fails on API runs).",
            file=sys.stderr,
        )
        return
    sa_b64 = base64.b64encode(sa_json.encode("utf-8")).decode("ascii")
    text = notebook_path.read_text(encoding="utf-8")
    if "__FIREBASE_SA_B64__" not in text:
        print("WARNING: placeholder __FIREBASE_SA_B64__ not found in notebook — skipping injection.", file=sys.stderr)
        return
    notebook_path.write_text(text.replace("__FIREBASE_SA_B64__", sa_b64), encoding="utf-8")
    print(f"✓ Injected Firebase credentials ({len(sa_b64)} b64 chars) into notebook")


def push_kernel(notebook_path: Path) -> None:
    folder = notebook_path.parent
    run(["kaggle", "kernels", "push", "-p", str(folder)])


def _try_download_outputs(slug: str, dest: Path) -> bool:
    """Attempt to download kernel outputs. Returns True if it produced files.

    `kaggle kernels output` exits 0 (or near-0) once the kernel has finished and
    its outputs are available — even if the kernel is still listed as 'running'
    by a broken status endpoint. We use this as a fallback signal when
    `kernels status` keeps returning 500s.
    """
    dest.mkdir(parents=True, exist_ok=True)
    try:
        subprocess.run(
            ["kaggle", "kernels", "output", slug, "-p", str(dest)],
            capture_output=True, text=True, check=True,
        )
    except subprocess.CalledProcessError:
        return False
    # Outputs are available only after the kernel completed. result.json is
    # written by the very last cell, so its presence is a strong "done" signal.
    return (dest / "result.json").exists()


def wait_for_completion(
    slug: str,
    timeout_min: int = 120,
    poll_sec: int = 60,
    output_dir: Path = Path("kaggle_outputs"),
    fallback_after_failures: int = 5,
) -> dict:
    """Poll `kaggle kernels status` until done or timeout.

    If the status endpoint returns errors (Kaggle's GetKernelSessionStatus has
    been flaky), after N consecutive failures we start probing for outputs
    instead. The kernel often runs and uploads to Firebase successfully even
    when the status API is broken — we don't want to declare failure in that
    case.
    """
    deadline = time.time() + timeout_min * 60
    last_status = None
    consecutive_failures = 0
    while time.time() < deadline:
        try:
            cp = run(["kaggle", "kernels", "status", slug])
            consecutive_failures = 0
        except subprocess.CalledProcessError as e:
            consecutive_failures += 1
            print(
                f"status check failed (transient, {consecutive_failures} in a row): {e.stderr}",
                file=sys.stderr,
            )
            if consecutive_failures >= fallback_after_failures:
                print(
                    f"status endpoint flaky — probing kernel outputs as fallback...",
                    file=sys.stderr,
                )
                if _try_download_outputs(slug, output_dir):
                    print("✓ kernel outputs available — treating as complete.")
                    return {"status": "complete", "raw": "(via output fallback)"}
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

    # Final fallback: try outputs one last time before declaring timeout.
    if _try_download_outputs(slug, output_dir):
        print("✓ kernel outputs available at deadline — treating as complete.")
        return {"status": "complete", "raw": "(via output fallback at timeout)"}
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
    inject_firebase_secret(args.notebook)
    push_kernel(args.notebook)

    print(f"Waiting up to {args.timeout_min} min for kernel run to finish...")
    result = wait_for_completion(
        args.slug,
        timeout_min=args.timeout_min,
        output_dir=args.output_dir,
    )
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
