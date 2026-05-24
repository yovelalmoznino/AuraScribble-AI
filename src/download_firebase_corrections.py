from __future__ import annotations

import argparse
import sys
from pathlib import Path

from google.cloud import storage

from upload_firebase import DEFAULT_BUCKET, _resolve_credentials

DEFAULT_PREFIX = "training_data/new/"


def download_corrections(
    *,
    local_dir: str | Path = "data/raw/training_data/new",
    bucket_name: str = DEFAULT_BUCKET,
    prefix: str = DEFAULT_PREFIX,
    credentials_path: str | None = None,
    include_processed: bool = False,
) -> int:
    """Download correction JSON blobs from Firebase Storage to a local folder tree."""
    creds = _resolve_credentials(credentials_path)
    if creds is None:
        raise RuntimeError("Firebase credentials missing (see upload_firebase.py help).")

    client = storage.Client(credentials=creds, project=creds.project_id)
    bucket = client.bucket(bucket_name)
    out_root = Path(local_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    prefixes = [prefix]
    if include_processed:
        prefixes.append("training_data/processed/")

    count = 0
    for pfx in prefixes:
        for blob in bucket.list_blobs(prefix=pfx):
            name = blob.name
            if name.endswith("/") or not name.endswith(".json"):
                continue
            rel = name
            if rel.startswith(pfx):
                rel = rel[len(pfx) :]
            dest = out_root / rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            blob.download_to_filename(str(dest))
            count += 1
            if count <= 5 or count % 100 == 0:
                print(f"  {name} -> {dest}")

    print(f"Downloaded {count} correction file(s) to {out_root}")
    return count


def download_checkpoint(
    *,
    local_path: str | Path = "models/checkpoint_best.pt",
    remote_path: str = "models/checkpoint_best.pt",
    bucket_name: str = DEFAULT_BUCKET,
    credentials_path: str | None = None,
) -> bool:
    creds = _resolve_credentials(credentials_path)
    if creds is None:
        return False
    client = storage.Client(credentials=creds, project=creds.project_id)
    blob = client.bucket(bucket_name).blob(remote_path)
    if not blob.exists():
        print(f"No remote checkpoint at gs://{bucket_name}/{remote_path}")
        return False
    dest = Path(local_path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    blob.download_to_filename(str(dest))
    print(f"Downloaded checkpoint -> {dest}")
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description="Download training corrections from Firebase Storage.")
    parser.add_argument("--local-dir", default="data/raw/training_data/new")
    parser.add_argument("--bucket", default=DEFAULT_BUCKET)
    parser.add_argument("--prefix", default=DEFAULT_PREFIX)
    parser.add_argument("--credentials", default=None)
    parser.add_argument("--include-processed", action="store_true")
    parser.add_argument("--checkpoint", action="store_true", help="Also download models/checkpoint_best.pt")
    args = parser.parse_args()

    try:
        n = download_corrections(
            local_dir=args.local_dir,
            bucket_name=args.bucket,
            prefix=args.prefix,
            credentials_path=args.credentials,
            include_processed=args.include_processed,
        )
        if args.checkpoint:
            download_checkpoint(credentials_path=args.credentials, bucket_name=args.bucket)
        if n == 0 and not args.checkpoint:
            print("No correction JSON files found under prefix.", file=sys.stderr)
    except Exception as exc:
        print(f"Download failed: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
