"""Shared CLI helpers for scripts."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
RESULTS = ROOT / "results"


def add_model_args(ap: argparse.ArgumentParser) -> None:
    ap.add_argument("--model", default=None, help="HF model id (default: $JQV_MODEL or Qwen/Qwen3-1.7B)")
    ap.add_argument("--dtype", default=None, help="float32|bfloat16|float16 (default: bfloat16 on GPU/MPS)")
    ap.add_argument("--device", default=None)


def load_rt(args):
    from jqv.model import load_runtime

    return load_runtime(args.model, args.device, args.dtype)


def dump_json(obj, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False))
    print(f"wrote {path}")


def slug(model_id: str) -> str:
    return model_id.split("/")[-1].lower()
