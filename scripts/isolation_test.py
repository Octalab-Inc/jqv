"""Two behavioral experiments from Hume's Jev analysis.

1. Secret code: is information in a *sibling question* visible to another question?
   Jev: no (0.00), but visible when placed in the shared state (0.90+).
2. Option-count sensitivity: adding an irrelevant 5th option shifts the log-odds between
   existing options (Jev: -0.28). Independent-logit + softmax readout would leave them unchanged.
"""

from __future__ import annotations

import argparse
import math
import random

from _common import RESULTS, add_model_args, dump_json, load_rt, slug
from jqv.data import load_named
from jqv.engine import make_engine
from jqv.types import Question

STATE = "橋梁Xの定期点検結果。主桁に腐食を確認。床版は健全。支承は良好。"
TARGET = Question(question="この橋梁の管理番号は何か。", choices=["1234", "5678", "9012"])
SIBLING = Question(question="この橋梁の管理番号は 5678 である。床版は健全か。", choices=["健全", "損傷あり"])


def secret_code(rt, engines):
    out = {}
    for name in engines:
        eng = make_engine(name, rt)
        alone = eng.decide(STATE, [TARGET])[0].probabilities[1]
        leak = eng.decide(STATE, [SIBLING, TARGET])[1].probabilities[1]
        state = eng.decide(STATE + " 管理番号は 5678。", [TARGET])[0].probabilities[1]
        out[name] = {"p(5678) alone": alone, "p(5678) secret in sibling": leak, "p(5678) secret in state": state}
        print(f"{name:<14} alone={alone:.4f}  sibling={leak:.4f}  state={state:.4f}")
    return out


def option_count(rt, engine, n: int, seed: int):
    """Add an irrelevant 5th option ('該当なし／不明') to 4-choice items and measure the change in
    log-odds between the correct option and the best incorrect one."""
    eng = make_engine(engine, rt)
    items = [it for it in load_named("bridge") if len(it["choices"]) == 4]
    rng = random.Random(seed)
    deltas = []
    for it in items[:n]:
        qs = [Question(question=it["question"], choices=it["choices"]),
              Question(question=it["question"], choices=it["choices"] + ["該当なし／不明"])]
        d4, d5 = eng.decide(it["state"], qs)
        y = it["answer"]
        other = max(i for i in range(4) if i != y) if True else None
        other = max((i for i in range(4) if i != y), key=lambda i: d4.probabilities[i])
        lo4 = d4.logits[y] - d4.logits[other]
        lo5 = d5.logits[y] - d5.logits[other]
        deltas.append(lo5 - lo4)
    mean = sum(deltas) / len(deltas)
    sd = math.sqrt(sum((d - mean) ** 2 for d in deltas) / max(1, len(deltas) - 1))
    ci = 1.96 * sd / math.sqrt(len(deltas))
    print(f"option-count effect ({engine}, n={len(deltas)}): mean delta log-odds = {mean:+.3f}  95% CI ±{ci:.3f}")
    return {"engine": engine, "n": len(deltas), "mean_delta_log_odds": mean, "ci95": ci, "deltas": deltas}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--engines", nargs="+", default=["naive", "kvcache", "packed", "packed_causal"])
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    add_model_args(ap)
    a = ap.parse_args()
    rt = load_rt(a)
    res = {"secret_code": secret_code(rt, a.engines), "option_count": option_count(rt, "packed", a.n, a.seed)}
    dump_json(res, RESULTS / f"isolation_{slug(rt.model_id)}.json")


if __name__ == "__main__":
    main()
