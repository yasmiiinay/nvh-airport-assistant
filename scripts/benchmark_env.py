"""Model load-time / memory / latency benchmark harness (Evidence Pack section A2).

Foundation stage: the harness and its measurement discipline exist and are
verified with the built-in 'noop' loader. Real model loaders are added in
Chat 03 -- the registry entries below raise NotImplementedError until then.
NEVER report a timing that was not produced by running this script.

Usage:
    python scripts/benchmark_env.py --model noop
    python scripts/benchmark_env.py --model noop --warmup 1 --runs 5
Output: appends one row to outputs/env_benchmark.csv
"""
from __future__ import annotations

import argparse
import csv
import platform
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from configs.settings import SETTINGS

try:
    import psutil
    _PROC = psutil.Process()

    def peak_rss_mb() -> float:
        return _PROC.memory_info().rss / (1024 ** 2)
except ImportError:  # harness still usable before pip install
    import resource

    def peak_rss_mb() -> float:
        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024


def _load_noop():
    """Baseline loader proving the harness works: trivial deterministic work."""
    return sum(i * i for i in range(200_000))


# TODO(Chat 03): implement these loaders. Each returns the loaded object.
#   clip      -> transformers CLIPModel + processor (SETTINGS.clip_model_id)
#   whisper   -> transformers ASR pipeline (SETTINGS.whisper_model_id)
#   minilm    -> sentence_transformers SentenceTransformer (SETTINGS.sentence_model_id)
#   easyocr   -> easyocr.Reader(['en'], gpu=False)  [only if SETTINGS.enable_ocr]
LOADERS = {
    "noop": _load_noop,
    "clip": None,
    "whisper": None,
    "minilm": None,
    "easyocr": None,
}


def detect_device() -> str:
    try:
        import torch
        mps = getattr(torch.backends, "mps", None)
        return "mps" if (mps and torch.backends.mps.is_available()) else "cpu"
    except ImportError:
        return "no-torch"


def benchmark(name: str, warmup: int, runs: int) -> dict:
    loader = LOADERS.get(name)
    if loader is None:
        raise NotImplementedError(
            f"Loader '{name}' is a Chat 03 task; only 'noop' runs at foundation stage.")
    rss_before = peak_rss_mb()
    for _ in range(max(0, warmup)):
        loader()
    timings = []
    for _ in range(max(1, runs)):
        start = time.perf_counter()
        loader()
        timings.append(time.perf_counter() - start)
    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "model": name,
        "device": detect_device(),
        "python": platform.python_version(),
        "machine": platform.machine(),
        "warmup_runs": warmup,
        "measured_runs": runs,
        "median_seconds": round(statistics.median(timings), 4),
        "min_seconds": round(min(timings), 4),
        "max_seconds": round(max(timings), 4),
        "rss_before_mb": round(rss_before, 1),
        "rss_after_mb": round(peak_rss_mb(), 1),
    }


def append_row(row: dict, csv_path: Path) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    new_file = not csv_path.exists()
    with open(csv_path, "a", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(row))
        if new_file:
            writer.writeheader()
        writer.writerow(row)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=sorted(LOADERS))
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--runs", type=int, default=5)
    args = parser.parse_args()
    row = benchmark(args.model, args.warmup, args.runs)
    out = SETTINGS.outputs_dir / "env_benchmark.csv"
    append_row(row, out)
    for k, v in row.items():
        print(f"{k:16s} {v}")
    print(f"\nappended to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
