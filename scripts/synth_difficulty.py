"""Accuracy by family / scenario / hops / question type for synthetic eval runs.

eval.py shuffles items with sample_split(seed); this script re-derives the same order from the dataset and joins the
saved logits (results/<tag>.npz) with each item's meta, so no change to eval.py is needed.

    uv run python scripts/synth_difficulty.py --model Qwen/Qwen3-32B --split dev [--engine packed] [--seed 0] [--md results/synth_difficulty.md]
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from _common import RESULTS, slug
from jqv.data import load_named, sample_split
from jqv.synth import FAMILIES


def rows_for(family: str, split: str, engine: str, model: str, seed: int, n: int, n_val: int, suffix: str = ""):
    tag = f"synth-{family}-{split}_{engine}_{slug(model)}{suffix}"
    npz = RESULTS / f"{tag}.npz"
    if not npz.exists():
        return None, tag
    d = np.load(npz)
    items = load_named(f"synth:{family}:{split}")
    val, test = sample_split(items, n or None, n_val, seed)
    ordered = val + test
    if len(ordered) != len(d["labels"]):
        raise SystemExit(f"{tag}: {len(ordered)} items vs {len(d['labels'])} rows; re-run eval.py with the same --n/--n-val/--seed")
    pred = d["logits"].argmax(-1)
    rows = []
    for it, y, yhat, z in zip(ordered, d["labels"], pred, d["logits"]):
        m = it["meta"]
        probs = np.exp(z - z.max()); probs = probs / probs.sum()
        labels = it["labels"]
        surface = m.get("surface_answer") or ""
        rows.append({"family": family, "scenario": m["scenario"], "hops": m["dependency_hops"], "depth": m["reasoning_depth"],
                     "qtype": it["qtype"], "correct": int(y == yhat), "p_max": float(probs.max()),
                     "took_surface": int(bool(surface) and labels[int(yhat)] == surface and surface != labels[int(y)]),
                     "has_surface": int(bool(surface) and surface != labels[int(y)])})
    return rows, tag


def table(rows, key):
    g = defaultdict(list)
    for r in rows:
        g[r[key]].append(r)
    out = []
    for k in sorted(g, key=lambda x: (str(type(x)), x)):
        rs = g[k]
        acc = sum(r["correct"] for r in rs) / len(rs)
        hs = [r for r in rs if r["has_surface"]]
        sr = sum(r["took_surface"] for r in hs) / len(hs) if hs else float("nan")
        out.append((k, len(rs), acc, sr))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--split", default="dev")
    ap.add_argument("--engine", default="packed")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n", type=int, default=0)
    ap.add_argument("--n-val", type=int, default=0)
    ap.add_argument("--suffix", default="", help="extra tag suffix, e.g. _14b-hardfam for --head-dir runs")
    ap.add_argument("--md", default=None, help="append the markdown tables to this file")
    ap.add_argument("--json", default=None)
    a = ap.parse_args()
    lines = [f"### {a.model} ({a.engine}, split={a.split}{a.suffix})", ""]
    all_rows, summary = [], {}
    for fam in FAMILIES:
        rows, tag = rows_for(fam, a.split, a.engine, a.model, a.seed, a.n, a.n_val, a.suffix)
        if rows is None:
            lines.append(f"- {fam}: no results ({tag}.npz missing)")
            continue
        all_rows += rows
        acc = sum(r["correct"] for r in rows) / len(rows)
        summary[fam] = {"n": len(rows), "accuracy": acc}
        lines.append(f"**{fam}**: n={len(rows)}, accuracy {acc:.3f}")
        lines.append("")
        for key, title in (("scenario", "scenario"), ("hops", "dependency_hops"), ("qtype", "type")):
            lines.append(f"| {title} | n | accuracy | surface-answer rate |")
            lines.append("|---|---:|---:|---:|")
            for k, n, ac, sr in table(rows, key):
                lines.append(f"| {k} | {n} | {ac:.3f} | {'-' if sr != sr else f'{sr:.2f}'} |")
            summary.setdefault(f"{fam}_by_{key}", {})
            for k, n, ac, sr in table(rows, key):
                summary[f"{fam}_by_{key}"][str(k)] = {"n": n, "accuracy": ac, "surface_rate": None if sr != sr else sr}
            lines.append("")
    if all_rows:
        lines.append("**all families by dependency_hops**")
        lines.append("")
        lines.append("| hops | n | accuracy | surface-answer rate |")
        lines.append("|---|---:|---:|---:|")
        for k, n, ac, sr in table(all_rows, "hops"):
            lines.append(f"| {k} | {n} | {ac:.3f} | {'-' if sr != sr else f'{sr:.2f}'} |")
        lines.append("")
    text = "\n".join(lines)
    print(text)
    if a.md:
        with open(a.md, "a") as fh:
            fh.write(text + "\n")
        print(f"appended to {a.md}")
    if a.json:
        Path(a.json).write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
