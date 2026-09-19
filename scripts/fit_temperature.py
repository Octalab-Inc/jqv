"""Fit temperature scaling on the val split of an eval.py cache and report test metrics before/after.

    uv run scripts/fit_temperature.py results/mmlu_packed_qwen3-1.7b.npz
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from _common import RESULTS, dump_json
from jqv.calibration import TemperatureScaler
from jqv.metrics import reliability_diagram, summary
from jqv.model import default_style_for
from jqv.prompt import prompt_hash


def provenance(npz_path: Path, d, is_val) -> dict:
    """Metadata for the temperature file: taken from eval.py's companion JSON, else derived."""
    meta = {}
    companion = npz_path.with_suffix(".json")
    if companion.exists():
        j = json.loads(companion.read_text())
        meta = {k: j.get(k) for k in ("model", "engine", "dataset", "dtype", "prompt_hash")}
    if not meta.get("model"):
        raise SystemExit(f"{npz_path}: no companion JSON with model/engine/dataset; rerun scripts/eval.py")
    if not meta.get("prompt_hash"):  # caches written before prompt_hash was recorded: the default style was used
        meta["prompt_hash"] = prompt_hash(default_style_for(meta["model"]))
    meta["n_val"] = int(is_val.sum())
    meta["choice_counts"] = sorted({int(c) for c in d["k"][is_val.numpy()]})
    return meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("npz", nargs="+", help="one or more eval caches (results/*.npz)")
    ap.add_argument("--out", default=None, help="temperature json path (default: results/<tag>_temperature.json)")
    a = ap.parse_args()

    for path in a.npz:
        path = Path(path)
        d = np.load(path)
        z, y, is_val = torch.tensor(d["logits"]), torch.tensor(d["labels"]), torch.tensor(d["is_val"])
        if is_val.sum() == 0:
            raise SystemExit(f"{path}: no val rows; rerun eval.py with --n-val > 0")
        ts = TemperatureScaler().fit(z[is_val], y[is_val])
        before = summary(z[~is_val].softmax(-1), y[~is_val])
        after = summary(ts.apply(z[~is_val]), y[~is_val])
        tag = path.stem
        print(f"\n{tag}: {ts}  (fit on {int(is_val.sum())} val rows, test on {int((~is_val).sum())})")
        print(f"{'metric':<18}{'before':>10}{'after':>10}")
        for key in ("accuracy", "nll", "brier", "ece", "mean_confidence"):
            print(f"{key:<18}{before[key]:>10.4f}{after[key]:>10.4f}")
        out = Path(a.out) if a.out else RESULTS / f"{tag}_temperature.json"
        ts.save(out, **provenance(path, d, is_val))
        print(f"wrote {out} ({ts})")
        dump_json({"temperature": ts.temperature, "meta": ts.meta, "before": before, "after": after},
                  RESULTS / f"{tag}_calibration.json")
        reliability_diagram(z[~is_val].softmax(-1), y[~is_val], RESULTS / f"{tag}_reliability_before.png", f"{tag} (T=1)")
        reliability_diagram(ts.apply(z[~is_val]), y[~is_val], RESULTS / f"{tag}_reliability_after.png", f"{tag} (T={ts.temperature:.2f})")


if __name__ == "__main__":
    main()
