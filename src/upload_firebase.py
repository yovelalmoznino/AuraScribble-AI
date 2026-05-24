from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

try:
    from google.cloud import storage
    from google.oauth2 import service_account
except ImportError as exc:
    raise SystemExit(
        "Missing dependency: pip install google-cloud-storage\n"
        f"Original error: {exc}"
    ) from exc


DEFAULT_BUCKET = "aurascribblr.firebasestorage.app"
DEFAULT_REMOTE = "models/latest_handwriting.onnx"

_REQUIRED_SA_FIELDS = ("type", "project_id", "private_key", "client_email", "token_uri")


def _validate_service_account_info(info: dict) -> None:
    if info.get("type") != "service_account":
        if "project_info" in info and "client" in info:
            raise ValueError(
                "This looks like google-services.json (Android app config), NOT a service account key.\n"
                "Get the correct file: Firebase Console → Project Settings → Service accounts → "
                "Generate new private key."
            )
        raise ValueError(
            f"Expected type='service_account', got {info.get('type')!r}. "
            "Use a Firebase Admin SDK private key JSON."
        )
    missing = [k for k in _REQUIRED_SA_FIELDS if k not in info or not info[k]]
    if missing:
        raise ValueError(
            f"Service account JSON is incomplete. Missing: {', '.join(missing)}.\n"
            "Download a fresh key: Firebase Console → Service accounts → Generate new private key."
        )


def _resolve_credentials(path: str | None) -> service_account.Credentials | None:
    """Load service account from file path, env var path, or inline JSON env."""
    inline = os.environ.get("FIREBASE_SERVICE_ACCOUNT_JSON")
    if inline:
        info = json.loads(inline.strip())
        _validate_service_account_info(info)
        return service_account.Credentials.from_service_account_info(info)

    cred_path = path or os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
    if cred_path and Path(cred_path).exists():
        info = json.loads(Path(cred_path).read_text(encoding="utf-8"))
        _validate_service_account_info(info)
        return service_account.Credentials.from_service_account_file(cred_path)

    default = Path("configs/firebase_service_account.json")
    if default.exists():
        info = json.loads(default.read_text(encoding="utf-8"))
        _validate_service_account_info(info)
        return service_account.Credentials.from_service_account_file(str(default))

    return None


def upload_file(
    local_path: Path,
    *,
    bucket_name: str,
    remote_path: str,
    credentials_path: str | None = None,
    content_type: str = "application/octet-stream",
) -> str:
    if not local_path.exists():
        raise FileNotFoundError(f"Local file not found: {local_path}")

    creds = _resolve_credentials(credentials_path)
    if creds is None:
        raise RuntimeError(
            "No Firebase credentials found. Use one of:\n"
            "  - configs/firebase_service_account.json\n"
            "  - GOOGLE_APPLICATION_CREDENTIALS=/path/to/sa.json\n"
            "  - FIREBASE_SERVICE_ACCOUNT_JSON='{...}' (Kaggle secret / CI)\n"
            "  - --credentials /path/to/sa.json"
        )

    client = storage.Client(credentials=creds, project=creds.project_id)
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(remote_path)
    blob.upload_from_filename(str(local_path), content_type=content_type)

    blob.reload()
    updated = blob.updated.isoformat() if blob.updated else "unknown"
    size_kb = local_path.stat().st_size / 1024
    print(f"Uploaded {local_path} ({size_kb:.1f} KB)")
    print(f"  gs://{bucket_name}/{remote_path}")
    print(f"  updated: {updated}")
    return f"gs://{bucket_name}/{remote_path}"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Upload handwriting ONNX model to Firebase Storage (OTA path)."
    )
    parser.add_argument("--local", default="output/model.onnx", help="Local ONNX file.")
    parser.add_argument("--remote", default=DEFAULT_REMOTE, help="Remote Storage path.")
    parser.add_argument(
        "--bucket",
        default=os.environ.get("FIREBASE_STORAGE_BUCKET", DEFAULT_BUCKET),
    )
    parser.add_argument("--credentials", default=None, help="Service account JSON path.")
    parser.add_argument("--vocab", default=None, help="Optional vocab.txt to upload.")
    parser.add_argument("--vocab-remote", default="models/latest_vocab.txt")
    args = parser.parse_args()

    local = Path(args.local)
    try:
        upload_file(
            local,
            bucket_name=args.bucket,
            remote_path=args.remote,
            credentials_path=args.credentials,
        )
        if args.vocab:
            vocab_path = Path(args.vocab)
            if vocab_path.exists():
                upload_file(
                    vocab_path,
                    bucket_name=args.bucket,
                    remote_path=args.vocab_remote,
                    credentials_path=args.credentials,
                    content_type="text/plain; charset=utf-8",
                )
            else:
                print(f"Warning: vocab not found: {vocab_path}", file=sys.stderr)
    except Exception as exc:
        print(f"Upload failed: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc

    print("Done. App OTA path:", args.remote)


if __name__ == "__main__":
    main()
