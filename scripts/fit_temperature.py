"""Fit temperature scaling on the val split of an eval.py cache and report test metrics before/after.

    uv run scripts/fit_temperature.py results/mmlu_packed_qwen3-1.7b.npz
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from _common import RESULTS, dump_json
from jqv.calibration import TemperatureScaler
from jqv.metrics import reliability_diagram, summary


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
        ts.save(out)
        dump_json({"temperature": ts.temperature, "before": before, "after": after}, RESULTS / f"{tag}_calibration.json")
        reliability_diagram(z[~is_val].softmax(-1), y[~is_val], RESULTS / f"{tag}_reliability_before.png", f"{tag} (T=1)")
        reliability_diagram(ts.apply(z[~is_val]), y[~is_val], RESULTS / f"{tag}_reliability_after.png", f"{tag} (T={ts.temperature:.2f})")


if __name__ == "__main__":
    main()
