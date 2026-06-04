"""DEPRECATED — see kaggle/KAGGLE_TRAIN_V8.ipynb and .github/workflows/main.yml.

The original auto_train.py tried to fine-tune on a tiny per-batch slice of
corrections using a stale CLI signature for train.py. That approach was
incompatible with the V8 hybrid (CTC+AR) training pipeline and would have
overwritten a working model with a broken one.

The new architecture:
  - GitHub Actions runs `scripts/count_corrections.py` daily
  - When ≥ 500 corrections accumulate, GitHub Actions triggers the Kaggle kernel
    `kaggle/KAGGLE_TRAIN_V8.ipynb` (free GPU) which:
      1. Pulls latest checkpoint from Firebase
      2. Downloads all queued corrections
      3. Re-runs prepare_raw to normalize them
      4. Trains 20 epochs of CTC+AR hybrid on the FULL train.jsonl
      5. Quality-gates the result (collapse / empty / CER)
      6. Uploads ONNX + checkpoint + vocab to Firebase
      7. Moves corrections from training_data/new/ → training_data/processed/

See `kaggle/SETUP.md` for one-time setup instructions.
"""

import sys


def main() -> None:
    print(__doc__, file=sys.stderr)
    print(
        "\nThis script is intentionally a no-op. Use the GitHub Actions workflow "
        "or run kaggle/KAGGLE_TRAIN_V8.ipynb on Kaggle directly.",
        file=sys.stderr,
    )
    sys.exit(2)


if __name__ == "__main__":
    main()
