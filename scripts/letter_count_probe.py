"""Probe: character-level counting ("how many 'r' in strawberry") through the decision readout.

    uv run python scripts/letter_count_probe.py --model Qwen/Qwen3-1.7B

Three direct formulations (score-type question over 0-5, with an empty state / the word spelled out / spelled out with
positions) and one jqv-style decomposition (one shared state, one yes/no branch per letter position, the count is taken
outside the model). Prints per-word predictions and the raw probability vectors; writes results/letter_count_<slug>.json.
"""

from __future__ import annotations

import argparse
import json

from _common import RESULTS, add_model_args, dump_json, load_rt, slug
from jqv.engine import make_engine
from jqv.types import Question

CASES = [("strawberry", "r", 3), ("mississippi", "s", 4), ("banana", "a", 3), ("raspberry", "r", 3), ("committee", "t", 2),
         ("bookkeeper", "e", 3), ("hello", "l", 2), ("necessary", "s", 1), ("occurrence", "c", 3), ("balloon", "o", 2),
         ("parallel", "l", 3), ("assessment", "s", 4), ("kayak", "k", 2), ("tennessee", "e", 4), ("cinnamon", "n", 3)]
CHOICES = [str(i) for i in range(6)]


def direct(eng, state_fn, title, out):
    ok = 0
    print(f"== {title}")
    for word, letter, ans in CASES:
        q = Question(question=f"How many times does the letter '{letter}' appear in the word '{word}'?", choices=CHOICES)
        p = eng.decide(state_fn(word), [q])[0].probabilities
        pred = max(range(6), key=lambda i: p[i])
        ok += pred == ans
        out.append({"mode": title, "word": word, "letter": letter, "truth": ans, "pred": pred, "p": [round(x, 3) for x in p]})
        print(f"  {word:12s} '{letter}' truth={ans} pred={pred} {'ok ' if pred == ans else 'NG '} p={[round(x, 2) for x in p]}")
    print(f"  accuracy {ok}/{len(CASES)}")
    return ok


def decomposed(eng, title, out):
    ok = 0
    print(f"== {title}")
    for word, letter, ans in CASES:
        state = f"The word is '{word}'. Spelled letter by letter with positions: " + ", ".join(f"{i + 1}={c}" for i, c in enumerate(word))
        qs = [Question(question=f"Is the letter at position {i + 1} of the word exactly '{letter}'?", choices=["yes", "no"]) for i in range(len(word))]
        yes = [d.probabilities[0] for d in eng.decide(state, qs)]
        count = sum(1 for p in yes if p > 0.5)
        ok += count == ans
        marks = "".join("Y" if p > 0.5 else "." for p in yes)
        out.append({"mode": title, "word": word, "letter": letter, "truth": ans, "pred": count, "positions": marks})
        print(f"  {word:12s} '{letter}' truth={ans} count={count} {'ok ' if count == ans else 'NG '} [{marks}]")
    print(f"  accuracy {ok}/{len(CASES)}")
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", default="packed")
    add_model_args(ap)
    a = ap.parse_args()
    rt = load_rt(a)
    eng = make_engine(a.engine, rt)
    out: list[dict] = []
    summary = {
        "plain": direct(eng, lambda w: "", "direct: plain question, empty state", out),
        "spelled": direct(eng, lambda w: f"The word is spelled letter by letter: {' '.join(w)}", "direct: state spells the word letter by letter", out),
        "positions": direct(eng, lambda w: f"The word is spelled letter by letter: {' '.join(w)}. Letter positions: " + ", ".join(f"{i + 1}={c}" for i, c in enumerate(w)),
                            "direct: state spells letters with positions", out),
        "decomposed": decomposed(eng, "decomposed: one yes/no branch per position on a shared spelled state, counted outside", out),
        "n": len(CASES), "model": rt.model_id, "engine": a.engine,
    }
    print(summary)
    dump_json({"summary": summary, "rows": out}, RESULTS / f"letter_count_{slug(rt.model_id)}.json")


if __name__ == "__main__":
    main()
