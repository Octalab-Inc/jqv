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


def chat_answer(rt, user: str, thinking: bool, max_new: int) -> tuple[str, int]:
    import re

    import torch

    tok, model = rt.tokenizer, rt.model
    text = tok.apply_chat_template([{"role": "user", "content": user}], tokenize=False, add_generation_prompt=True, enable_thinking=thinking)
    enc = tok(text, return_tensors="pt").to(rt.device)
    with torch.no_grad():
        out = model.generate(**enc, max_new_tokens=max_new, do_sample=False)
    gen = out[0, enc["input_ids"].shape[1]:]
    ans = tok.decode(gen, skip_special_tokens=True)
    tail = ans.split("</think>")[-1]
    nums = re.findall(r"\b[0-9]\b", tail)
    return (nums[-1] if nums else "?"), int(gen.shape[0])


def generation_baseline(rt, n_words: int, max_new: int, out):
    """Ordinary chat generation for the first n_words cases, thinking off and on: the answer the model writes at the end."""
    res = {}
    for thinking in (False, True):
        ok = 0
        title = f"chat generation, thinking {'ON' if thinking else 'OFF'}"
        print(f"== {title} (first {n_words} words)")
        for word, letter, ans in CASES[:n_words]:
            user = f"How many times does the letter '{letter}' appear in the word '{word}'?\nchoices: 0 / 1 / 2 / 3 / 4 / 5"
            pred, n_tok = chat_answer(rt, user, thinking, max_new)
            ok += pred == str(ans)
            out.append({"mode": title, "word": word, "letter": letter, "truth": ans, "pred": pred, "tokens": n_tok})
            print(f"  {word:12s} '{letter}' truth={ans} answer={pred} {'ok ' if pred == str(ans) else 'NG '} ({n_tok} tokens)")
        print(f"  accuracy {ok}/{n_words}")
        res["chat_thinking_on" if thinking else "chat_thinking_off"] = ok
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", default="packed")
    ap.add_argument("--chat-words", type=int, default=5, help="words to also answer by ordinary generation (0 = skip)")
    ap.add_argument("--chat-max-new", type=int, default=900)
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
    if a.chat_words:
        summary.update(generation_baseline(rt, a.chat_words, a.chat_max_new, out))
        summary["chat_words"] = a.chat_words
    print(summary)
    dump_json({"summary": summary, "rows": out}, RESULTS / f"letter_count_{slug(rt.model_id)}.json")


if __name__ == "__main__":
    main()
