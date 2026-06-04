"""Count new corrections in Firebase. Used by GitHub Actions to decide whether to retrain.

Prints a single line: COUNT=<n>

Reads FIREBASE_SERVICE_ACCOUNT secret from env var (GH Actions writes it to a file
beforehand or passes it directly).
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import firebase_admin
from firebase_admin import credentials, storage


def count_new(bucket_name: str, prefix: str = "training_data/new/") -> int:
    bucket = storage.bucket(bucket_name)
    n = 0
    for blob in bucket.list_blobs(prefix=prefix):
        name = blob.name
        if name.endswith("/") or not name.endswith(".json"):
            continue
        n += 1
    return n


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--credentials", default="firebase-service-account.json")
    ap.add_argument(
        "--bucket",
        default=os.environ.get("FIREBASE_BUCKET", "aurascribblr.firebasestorage.app"),
    )
    ap.add_argument("--prefix", default="training_data/new/")
    ap.add_argument(
        "--threshold",
        type=int,
        default=int(os.environ.get("RETRAIN_THRESHOLD", "500")),
    )
    args = ap.parse_args()

    cred_path = Path(args.credentials)
    if not cred_path.exists():
        print(f"ERROR: missing credentials at {cred_path}", file=sys.stderr)
        sys.exit(2)

    if not firebase_admin._apps:
        firebase_admin.initialize_app(credentials.Certificate(str(cred_path)))

    n = count_new(args.bucket, args.prefix)
    print(f"COUNT={n}")
    print(f"THRESHOLD={args.threshold}")
    print(f"SHOULD_RETRAIN={'true' if n >= args.threshold else 'false'}")


if __name__ == "__main__":
    main()
