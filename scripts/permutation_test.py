"""Permutation experiments: how much do the returned probabilities depend on things that should not matter?

  --mode label : keep option order fixed, cyclically shift which LETTER each position carries
                 -> measures the letter-token prior (A/B/C/D as vocabulary tokens)
  --mode order : keep letters A,B,C,D in place, cyclically shift the option TEXTS
                 -> measures the listwise position/order effect
  --mode fifth : append an irrelevant 5th option; measure the shift in log-odds between the correct
                 option and the strongest wrong one (Hume measured -0.28 on Jev)

    uv run scripts/permutation_test.py --mode label --n 300
"""

from __future__ import annotations

import argparse
import itertools
import math
import random

import numpy as np

from _common import RESULTS, add_model_args, dump_json, load_rt, slug
from jqv.data import load_named
from jqv.engine import make_engine
from jqv.prompt import LETTERS
from jqv.types import Question

FIFTH_OPTION = "該当なし／不明"


def decide_grouped(engine, items, group_size=32):
    """items: list of (state, Question, payload). Returns decisions in the same order, sharing states."""
    out = [None] * len(items)
    by_state: dict[str, list[int]] = {}
    for i, (state, _q, _p) in enumerate(items):
        by_state.setdefault(state, []).append(i)
    done = 0
    for state, idxs in by_state.items():
        for s in range(0, len(idxs), group_size):
            chunk = idxs[s : s + group_size]
            ds = engine.decide(state, [items[i][1] for i in chunk])
            for i, d in zip(chunk, ds):
                out[i] = d
            done += len(chunk)
            print(f"\r{done}/{len(items)}", end="", flush=True)
    print()
    return out


def build(mode: str, data: list[dict]):
    """Return (jobs, meta). jobs: (state, Question, (item_idx, variant, pos_of_correct, sem_of_pos))
    sem_of_pos[pos] = semantic (original) choice index shown at that position."""
    jobs = []
    for it_idx, it in enumerate(data):
        k = len(it["choices"])
        for s in range(k):
            if mode == "label":
                labels = [LETTERS[(i + s) % k] for i in range(k)]
                q = Question(question=it["question"], choices=it["choices"], labels=labels)
                sem = list(range(k))  # position i still shows original choice i
            elif mode == "order":
                sem = [(i + s) % k for i in range(k)]  # position i shows original choice (i+s)%k
                q = Question(question=it["question"], choices=[it["choices"][j] for j in sem])
            else:
                raise ValueError(mode)
            jobs.append((it["state"], q, (it_idx, s, sem.index(it["answer"]), sem)))
    return jobs


def analyze(mode: str, data: list[dict], jobs, decisions):
    """Per item: probs over semantic choices for each variant."""
    k_of = {i: len(it["choices"]) for i, it in enumerate(data)}
    per_item: dict[int, dict[int, np.ndarray]] = {}
    slot_prob = {}  # letter (label mode) or position (order mode) -> list of probabilities (content-averaged)
    for (state, q, (it_idx, s, pos_correct, sem)), d in zip(jobs, decisions):
        p = np.array(d.probabilities)
        sem_p = np.zeros(k_of[it_idx])
        for pos, j in enumerate(sem):
            sem_p[j] = p[pos]
        per_item.setdefault(it_idx, {})[s] = sem_p
        for pos in range(k_of[it_idx]):
            key = (q.labels or list(LETTERS[: k_of[it_idx]]))[pos] if mode == "label" else pos
            slot_prob.setdefault(key, []).append(p[pos])
    diffs, consistent, acc_by_variant = [], 0, {}
    for it_idx, variants in per_item.items():
        y = data[it_idx]["answer"]
        pc = [variants[s][y] for s in sorted(variants)]
        diffs.append(np.mean([abs(a - b) for a, b in itertools.combinations(pc, 2)]) if len(pc) > 1 else 0.0)
        argmaxes = {int(variants[s].argmax()) for s in variants}
        consistent += len(argmaxes) == 1
        for s, sp in variants.items():
            acc_by_variant.setdefault(s, []).append(int(sp.argmax() == y))
    n = len(per_item)
    slot_table = {str(key): float(np.mean(v)) for key, v in sorted(slot_prob.items(), key=lambda kv: str(kv[0]))}
    return {
        "mode": mode,
        "n_items": n,
        "slot_prior": slot_table,  # label mode: mean prob per letter; order mode: mean prob per position
        "mean_abs_diff_p_correct": float(np.mean(diffs)),
        "argmax_consistent_fraction": consistent / n,
        "accuracy_by_variant": {str(s): float(np.mean(v)) for s, v in sorted(acc_by_variant.items())},
        "accuracy_min_max": [min(float(np.mean(v)) for v in acc_by_variant.values()),
                             max(float(np.mean(v)) for v in acc_by_variant.values())],
    }


def fifth_option(engine, data: list[dict]):
    jobs = []
    for it_idx, it in enumerate(data):
        if len(it["choices"]) != 4:
            continue
        jobs.append((it["state"], Question(question=it["question"], choices=it["choices"]), (it_idx, 4)))
        jobs.append((it["state"], Question(question=it["question"], choices=it["choices"] + [FIFTH_OPTION]), (it_idx, 5)))
    ds = decide_grouped(engine, jobs)
    by_item: dict[int, dict[int, list[float]]] = {}
    for (_s, _q, (it_idx, k)), d in zip(jobs, ds):
        by_item.setdefault(it_idx, {})[k] = d.logits
    deltas, fifth_mass, flips = [], [], 0
    for it_idx, r in by_item.items():
        y = data[it_idx]["answer"]
        z4, z5 = r[4], r[5]
        other = max((i for i in range(4) if i != y), key=lambda i: z4[i])
        deltas.append((z5[y] - z5[other]) - (z4[y] - z4[other]))
        p5 = np.exp(np.array(z5) - np.max(z5)); p5 /= p5.sum()
        fifth_mass.append(float(p5[4]))
        flips += int(np.argmax(z4) != np.argmax(z5[:4]))
    mean = float(np.mean(deltas)); sd = float(np.std(deltas, ddof=1)); ci = 1.96 * sd / math.sqrt(len(deltas))
    return {"mode": "fifth", "n_items": len(deltas), "mean_delta_log_odds": mean, "ci95": ci, "sd": sd,
            "mean_prob_on_fifth": float(np.mean(fifth_mass)), "argmax_flip_fraction": flips / len(deltas),
            "deltas": deltas}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["label", "order", "fifth"], required=True)
    ap.add_argument("--datasets", nargs="+", default=["mmlu", "bridge"])
    ap.add_argument("--n", type=int, default=300, help="items per dataset (0 = all); bridge is always all")
    ap.add_argument("--engine", default="packed")
    ap.add_argument("--seed", type=int, default=0)
    add_model_args(ap)
    a = ap.parse_args()
    rt = load_rt(a)
    engine = make_engine(a.engine, rt)
    results = {}
    for name in a.datasets:
        data = load_named(name)
        if name != "bridge" and a.n:
            rng = random.Random(a.seed)
            data = rng.sample(data, min(a.n, len(data)))
        print(f"== {name}: {len(data)} items, mode={a.mode}")
        if a.mode == "fifth":
            res = fifth_option(engine, data)
        else:
            jobs = build(a.mode, data)
            res = analyze(a.mode, data, jobs, decide_grouped(engine, jobs))
        res.update(dataset=name, engine=a.engine, model=rt.model_id)
        results[name] = res
        print({k: v for k, v in res.items() if k != "deltas"})
    dump_json(results, RESULTS / f"permutation_{a.mode}_{slug(rt.model_id)}.json")


if __name__ == "__main__":
    main()
