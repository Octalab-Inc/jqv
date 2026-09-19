"""Cross-dataset temperature transfer.

Fit T on the val split of a *source* dataset and apply it to the test split of every *target*
dataset. Uses the logits cached by eval.py (results/<dataset>_<engine>_<model>.npz), so no inference.

    uv run scripts/transfer_temperature.py            # sources: mmlu, jmmlu, mmlu+jmmlu; targets: mmlu, jmmlu, bridge

Rows: T=1 (no calibration), one row per source, and an oracle row per target (T fitted on the target's own
test rows: an optimistic upper bound, the only option for bridge which has no val split).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from _common import RESULTS, dump_json
from jqv.calibration import TemperatureScaler
from jqv.metrics import reliability_diagram, summary


def load(tag: str):
    d = np.load(RESULTS / f"{tag}.npz")
    return torch.tensor(d["logits"]), torch.tensor(d["labels"]), torch.tensor(d["is_val"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", default="packed")
    ap.add_argument("--model-slug", default="qwen3-1.7b")
    ap.add_argument("--sources", nargs="+", default=["mmlu", "jmmlu"])
    ap.add_argument("--targets", nargs="+", default=["mmlu", "jmmlu", "bridge"])
    ap.add_argument("--diagrams", action="store_true", help="also write reliability diagrams for transfer cells")
    a = ap.parse_args()

    data = {n: load(f"{n}_{a.engine}_{a.model_slug}") for n in set(a.sources) | set(a.targets)}
    # temperatures: per source (val rows), pooled over all sources, and per-target oracle (test rows)
    temps: dict[str, float] = {}
    for src in a.sources:
        z, y, v = data[src]
        if v.sum() == 0:
            raise SystemExit(f"{src}: no val rows to fit on")
        temps[src] = TemperatureScaler().fit(z[v], y[v]).temperature
    if len(a.sources) > 1:
        z = torch.cat([data[s][0][data[s][2]] for s in a.sources])
        y = torch.cat([data[s][1][data[s][2]] for s in a.sources])
        temps["+".join(a.sources)] = TemperatureScaler().fit(z, y).temperature
    for tgt in a.targets:
        z, y, v = data[tgt]
        temps[f"oracle:{tgt}"] = TemperatureScaler().fit(z[~v], y[~v]).temperature

    rows = []
    def cell(name: str, T: float, tgt: str):
        z, y, v = data[tgt]
        p = (z[~v] / T).softmax(-1)
        m = summary(p, y[~v])
        return {"source": name, "target": tgt, "temperature": T, **m}

    for tgt in a.targets:
        rows.append(cell("none (T=1)", 1.0, tgt))
        for name, T in temps.items():
            if name.startswith("oracle:") and name != f"oracle:{tgt}":
                continue
            rows.append(cell(name, T, tgt))
            if a.diagrams and name in a.sources and name != tgt:
                z, y, v = data[tgt]
                reliability_diagram((z[~v] / T).softmax(-1), y[~v], RESULTS / f"transfer_{name}_to_{tgt}_reliability.png",
                                    f"T({name})={T:.2f} applied to {tgt}")

    # print one table per target
    for tgt in a.targets:
        n = int((~data[tgt][2]).sum())
        print(f"\n### target = {tgt} (test n={n}, accuracy {rows[[r['target'] for r in rows].index(tgt)]['accuracy']:.3f})")
        print(f"{'T fitted on':<16}{'T':>8}{'ECE':>8}{'NLL':>8}{'Brier':>8}{'mean conf':>11}")
        for r in rows:
            if r["target"] == tgt:
                print(f"{r['source']:<16}{r['temperature']:>8.2f}{r['ece']:>8.3f}{r['nll']:>8.3f}{r['brier']:>8.3f}{r['mean_confidence']:>11.3f}")
    dump_json({"temperatures": temps, "rows": rows}, RESULTS / f"transfer_{a.model_slug}.json")


if __name__ == "__main__":
    main()
