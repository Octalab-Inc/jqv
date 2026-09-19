"""Accuracy + calibration evaluation.

    uv run scripts/eval.py --dataset mmlu --engine packed --n 1200 --n-val 400

Items are grouped (`--group-size`) into one decide() call sharing the (empty or common) state,
which is the Jev-style "many questions, one state" usage. Raw choice logits are cached to
results/<dataset>_<engine>_<model>.npz for fit_temperature.py.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch

from _common import RESULTS, add_model_args, dump_json, load_rt, slug
from jqv.data import load_named, sample_split
from jqv.engine import make_engine
from jqv.metrics import summary
from jqv.types import Question


def run(engine, items, group_size: int):
    """Decide all items, grouping by state so a shared state is prefilled once. Results are returned in the
    ORIGINAL item order (val first, then test), so caches from different settings line up row by row."""
    results = [None] * len(items)
    by_state: dict[str, list[int]] = {}
    for idx, it in enumerate(items):
        by_state.setdefault(it["state"], []).append(idx)
    t0 = time.time()
    n_done = 0
    for state, idxs in by_state.items():
        for i in range(0, len(idxs), group_size):
            chunk = idxs[i : i + group_size]
            ds = engine.decide(state, [Question(question=items[j]["question"], choices=items[j]["choices"]) for j in chunk])
            for j, d in zip(chunk, ds):
                results[j] = d.logits
            n_done += len(chunk)
            print(f"\r{n_done}/{len(items)}  {n_done / (time.time() - t0):.1f} q/s", end="", flush=True)
    print()
    ks = [len(it["choices"]) for it in items]
    kmax = max(ks)
    z = np.full((len(items), kmax), -np.inf, dtype=np.float32)
    for i, row in enumerate(results):
        z[i, : len(row)] = row
    return z, np.array([it["answer"] for it in items]), np.array(ks), time.time() - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="mmlu", help="mmlu | jmmlu | bridge | path/to/file.jsonl")
    ap.add_argument("--engine", default="packed")
    ap.add_argument("--n", type=int, default=1200, help="total items (val + test); 0 = all")
    ap.add_argument("--n-val", type=int, default=400, help="items reserved for temperature fitting")
    ap.add_argument("--group-size", type=int, default=32)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--head-dir", default=None, help="for --engine pointer|slot: results/train/<run>/best")
    ap.add_argument("--shots", type=int, default=0, help="k exemplars in the shared state (MMLU dev)")
    ap.add_argument("--shots-mode", choices=["subject", "fixed"], default="subject",
                    help="subject: same-subject exemplars (one state per subject); fixed: one cross-subject state for all")
    ap.add_argument("--perm-avg", action="store_true", help="average over all cyclic rotations of the option order")
    add_model_args(ap)
    a = ap.parse_args()

    rt = load_rt(a)
    kw = {"head_dir": a.head_dir} if a.head_dir else {}
    eng = make_engine(a.engine, rt, perm_avg=a.perm_avg, **kw)
    items = load_named(a.dataset)
    val, test = sample_split(items, a.n or None, a.n_val, a.seed)
    if a.shots:
        from jqv.fewshot import fewshot_state, fixed_state

        for it in val + test:
            if it.get("state"):  # datasets with their own state (bridge): few-shot not applicable
                continue
            it["state"] = fixed_state(a.shots) if a.shots_mode == "fixed" else fewshot_state(it.get("subject"), a.shots)
    print(f"{a.dataset}: {len(val)} val / {len(test)} test  engine={a.engine} model={rt.model_id} dtype={rt.dtype} "
          f"shots={a.shots}{'/' + a.shots_mode if a.shots else ''} perm_avg={a.perm_avg}")

    z, y, k, secs = run(eng, val + test, a.group_size)
    is_val = np.zeros(len(y), dtype=bool)
    is_val[: len(val)] = True
    probs = torch.tensor(z).softmax(-1).numpy()
    m = summary(probs[~is_val], y[~is_val])
    m.update(engine=a.engine, model=rt.model_id, dtype=str(rt.dtype), dataset=a.dataset, seconds=secs,
             questions_per_sec=len(y) / secs, prompt_hash=rt.prompt.hash, n_val=len(val),
             choice_counts=sorted({int(c) for c in k}), shots=a.shots, shots_mode=a.shots_mode if a.shots else None,
             perm_avg=a.perm_avg)
    if a.engine == "generate":
        m["accuracy_note"] = "generate engine: probabilities are one-hot, calibration metrics not meaningful"
    print(m)

    tag = (f"{a.dataset}_{a.engine}_{slug(rt.model_id)}" + (f"_{Path(a.head_dir).parent.name}" if a.head_dir else "")
           + (f"_shots{a.shots}{'fixed' if a.shots_mode == 'fixed' else ''}" if a.shots else "") + ("_permavg" if a.perm_avg else ""))
    np.savez(RESULTS / f"{tag}.npz", logits=z, labels=y, k=k, is_val=is_val)
    dump_json(m, RESULTS / f"{tag}.json")


if __name__ == "__main__":
    main()
