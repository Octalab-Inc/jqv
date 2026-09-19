"""Accuracy + calibration evaluation.

    uv run scripts/eval.py --dataset mmlu --engine packed --n 1200 --n-val 400

Items are grouped (`--group-size`) into one decide() call sharing the (empty or common) state,
which is the Jev-style "many questions, one state" usage. Raw choice logits are cached to
results/<dataset>_<engine>_<model>.npz for fit_temperature.py.
"""

from __future__ import annotations

import argparse
import time

import numpy as np
import torch

from _common import RESULTS, add_model_args, dump_json, load_rt, slug
from jqv.data import load_named, sample_split
from jqv.engine import make_engine
from jqv.metrics import summary
from jqv.types import Question


def run(engine, items, group_size: int):
    logits, labels, ks = [], [], []
    # group by state so that a shared state is prefilled once
    by_state: dict[str, list] = {}
    for it in items:
        by_state.setdefault(it["state"], []).append(it)
    t0 = time.time()
    n_done = 0
    for state, group in by_state.items():
        for i in range(0, len(group), group_size):
            chunk = group[i : i + group_size]
            ds = engine.decide(state, [Question(question=c["question"], choices=c["choices"]) for c in chunk])
            for c, d in zip(chunk, ds):
                logits.append(d.logits)
                labels.append(c["answer"])
                ks.append(len(c["choices"]))
            n_done += len(chunk)
            print(f"\r{n_done}/{len(items)}  {n_done / (time.time() - t0):.1f} q/s", end="", flush=True)
    print()
    kmax = max(ks)
    z = np.full((len(logits), kmax), -np.inf, dtype=np.float32)
    for i, row in enumerate(logits):
        z[i, : len(row)] = row
    return z, np.array(labels), np.array(ks), time.time() - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="mmlu", help="mmlu | jmmlu | bridge | path/to/file.jsonl")
    ap.add_argument("--engine", default="packed")
    ap.add_argument("--n", type=int, default=1200, help="total items (val + test); 0 = all")
    ap.add_argument("--n-val", type=int, default=400, help="items reserved for temperature fitting")
    ap.add_argument("--group-size", type=int, default=32)
    ap.add_argument("--seed", type=int, default=0)
    add_model_args(ap)
    a = ap.parse_args()

    rt = load_rt(a)
    eng = make_engine(a.engine, rt)
    items = load_named(a.dataset)
    val, test = sample_split(items, a.n or None, a.n_val, a.seed)
    print(f"{a.dataset}: {len(val)} val / {len(test)} test  engine={a.engine} model={rt.model_id} dtype={rt.dtype}")

    z, y, k, secs = run(eng, val + test, a.group_size)
    is_val = np.zeros(len(y), dtype=bool)
    is_val[: len(val)] = True
    probs = torch.tensor(z).softmax(-1).numpy()
    m = summary(probs[~is_val], y[~is_val])
    m.update(engine=a.engine, model=rt.model_id, dtype=str(rt.dtype), dataset=a.dataset, seconds=secs,
             questions_per_sec=len(y) / secs)
    if a.engine == "generate":
        m["accuracy_note"] = "generate engine: probabilities are one-hot, calibration metrics not meaningful"
    print(m)

    tag = f"{a.dataset}_{a.engine}_{slug(rt.model_id)}"
    np.savez(RESULTS / f"{tag}.npz", logits=z, labels=y, k=k, is_val=is_val)
    dump_json(m, RESULTS / f"{tag}.json")


if __name__ == "__main__":
    main()
