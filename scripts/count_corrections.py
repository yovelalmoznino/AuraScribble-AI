"""Count new corrections in Firebase. Used by GitHub Actions to decide whether to retrain.

Prints a single line: COUNT=<n>

Credential resolution, in priority order:
  1. --credentials PATH (or default firebase-service-account.json if it exists) — CI path.
  2. GOOGLE_APPLICATION_CREDENTIALS env var.
  3. Application Default Credentials (gcloud auth application-default login) — local dev path.
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
    env_cred = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")

    if not firebase_admin._apps:
        if cred_path.exists():
            firebase_admin.initialize_app(credentials.Certificate(str(cred_path)))
        elif env_cred and Path(env_cred).exists():
            firebase_admin.initialize_app(credentials.Certificate(env_cred))
        else:
            # Application Default Credentials (gcloud, or GCE/GKE metadata server in CI).
            try:
                firebase_admin.initialize_app(options={"projectId": "aurascribblr"})
            except Exception as exc:
                print(
                    f"ERROR: no credentials. Pass --credentials PATH, set "
                    f"GOOGLE_APPLICATION_CREDENTIALS, or run "
                    f"`gcloud auth application-default login`. ({exc})",
                    file=sys.stderr,
                )
                sys.exit(2)

    n = count_new(args.bucket, args.prefix)
    print(f"COUNT={n}")
    print(f"THRESHOLD={args.threshold}")
    print(f"SHOULD_RETRAIN={'true' if n >= args.threshold else 'false'}")


if __name__ == "__main__":
    main()
