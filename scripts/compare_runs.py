"""Paired comparison of eval.py caches on the same test items: accuracy with binomial 95% CI, paired
difference vs. a baseline with bootstrap CI and an exact McNemar test, and selective accuracy
(coverage / accuracy when acting only if the calibrated p_max >= threshold).

    uv run scripts/compare_runs.py --dataset mmlu \
        --runs "B (vocab)=packed_qwen3-1.7b" "slot+LoRA=slot_qwen3-1.7b_slot_lora"

Each run tag names results/<dataset>_<tag>.npz (+ _temperature.json for the calibrated columns; T=1 if absent).
The first run is the baseline. All caches must share the same test items (same --seed / --n / --n-val in eval.py).
"""

from __future__ import annotations

import argparse
import json
from math import comb

import numpy as np

from _common import RESULTS, dump_json


def load(dataset: str, tag: str):
    d = np.load(RESULTS / f"{dataset}_{tag}.npz")
    tpath = RESULTS / f"{dataset}_{tag}_temperature.json"
    T = json.loads(tpath.read_text())["temperature"] if tpath.exists() else 1.0
    z, y, v = d["logits"], d["labels"], d["is_val"]
    z, y = z[~v], y[~v]

    def softmax(zz):
        zz = zz - zz.max(1, keepdims=True)
        e = np.exp(zz)
        return e / e.sum(1, keepdims=True)

    return {"y": y, "p_raw": softmax(z), "p_cal": softmax(z / T), "T": T}


def mcnemar_exact(b: int, c: int) -> float:
    tot = b + c
    if tot == 0:
        return 1.0
    m = min(b, c)
    return min(1.0, 2 * sum(comb(tot, i) for i in range(m + 1)) / 2**tot)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="mmlu")
    ap.add_argument("--runs", nargs="+", required=True, help='"label=tag" pairs; first is the baseline')
    ap.add_argument("--thresholds", type=float, nargs="+", default=[0.5, 0.7, 0.9])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)
    runs = [(s.split("=", 1)[0], s.split("=", 1)[1]) for s in a.runs]
    R = {label: load(a.dataset, tag) for label, tag in runs}
    base_label = runs[0][0]
    y0 = R[base_label]["y"]
    if not all((r["y"] == y0).all() for r in R.values()):
        raise SystemExit("test items differ between caches; rerun eval.py with the same --seed/--n/--n-val")
    n = len(y0)
    base_c = R[base_label]["p_raw"].argmax(1) == y0
    rows = []
    print(f"### {a.dataset} test n={n}: accuracy ±95% CI, paired Δ vs {base_label} [bootstrap 95% CI], McNemar p")
    print(f"| run | accuracy | Δ vs baseline | discordant (win/lose) | McNemar p |")
    print("|---|---:|---:|---:|---:|")
    for label, r in R.items():
        c = r["p_raw"].argmax(1) == r["y"]
        acc = c.mean()
        se = np.sqrt(acc * (1 - acc) / n)
        d = c.astype(int) - base_c.astype(int)
        boots = np.array([rng.choice(d, n).mean() for _ in range(2000)])
        lo, hi = np.percentile(boots, [2.5, 97.5])
        win, lose = int((c & ~base_c).sum()), int((~c & base_c).sum())
        p = mcnemar_exact(win, lose)
        rows.append({"run": label, "accuracy": acc, "ci95": 1.96 * se, "delta": d.mean(), "delta_ci": [lo, hi],
                     "win": win, "lose": lose, "mcnemar_p": p, "T": r["T"]})
        print(f"| {label} | {acc:.3f} ±{1.96 * se:.3f} | {d.mean():+.3f} [{lo:+.3f}, {hi:+.3f}] | {win}/{lose} | {p:.3f} |")
    print(f"\n### selective accuracy with calibrated p (T from val): coverage / accuracy when p_max >= threshold")
    print("| run | " + " | ".join(f"p ≥ {t}" for t in a.thresholds) + " |")
    print("|---|" + "---|" * len(a.thresholds))
    for label, r in R.items():
        conf, pred = r["p_cal"].max(1), r["p_cal"].argmax(1)
        c = pred == r["y"]
        cells, rec = [], {}
        for t in a.thresholds:
            m = conf >= t
            cov, acc = m.mean(), (c[m].mean() if m.any() else float("nan"))
            cells.append(f"{cov:.0%} を {acc:.2f}" if m.any() else "0%")
            rec[str(t)] = {"coverage": float(cov), "accuracy": float(acc) if m.any() else None}
        next(x for x in rows if x["run"] == label)["selective"] = rec
        print(f"| {label} | " + " | ".join(cells) + " |")
    out = a.out or (RESULTS / f"compare_{a.dataset}.json")
    dump_json({"dataset": a.dataset, "n": n, "baseline": base_label, "rows": rows}, RESULTS / out if not str(out).startswith("/") else out)


if __name__ == "__main__":
    main()
