"""TypeSafe-compatible wire format (`POST /v1/systemone`) on top of jqv engines.

Request  {"state": str, "model": str, "questions": {name: {"type": "choice"|"noul"|"score",
                                                          "instructions": str, "criteria": dict | list | None}}}
Response {"answers": {name: answer}, "usage": {"input_tokens": n, "output_tokens": 0}, "model": str}
  choice  answer = {"type": "choice", "choice": label, "probabilities": {label: p}}   labels = criteria keys (in order)
  noul    answer = {"type": "noul", "noul": p_yes}                                   options shown as "no", "yes"
  score   answer = {"type": "score", "probabilities": {"0": p, "1": p, ...}, "score": expected level}
                                                                                    levels = criteria list (in order)
Every question in one request shares the state, so they are decided in one call (shared-prefix engines prefill
the state once). This is the protocol JevBench's `typesafe` adapter speaks (github.com/fstandhartinger/jevbench).
"""

from __future__ import annotations

from jqv.types import Question

NOUL_LABELS = ["no", "yes"]  # presentation order; the answer is p("yes") regardless


class SystemOneError(ValueError):
    pass


def build_question(name: str, q: dict) -> tuple[Question, dict]:
    """Translate one wire-format question into a jqv Question plus the mapping needed to answer it."""
    if not isinstance(q, dict) or "type" not in q:
        raise SystemOneError(f"question {name!r}: missing type")
    qtype = q["type"]
    instructions = (q.get("instructions") or "").strip()
    criteria = q.get("criteria")
    if qtype == "choice":
        if not isinstance(criteria, dict) or len(criteria) < 2:
            raise SystemOneError(f"question {name!r}: choice needs a criteria dict with >= 2 labels")
        labels = list(criteria.keys())
        choices = [f"{label}: {criteria[label]}" if criteria[label] else str(label) for label in labels]
        return Question(question=instructions, choices=choices), {"type": "choice", "labels": labels}
    if qtype == "noul":
        desc = criteria if isinstance(criteria, dict) else {}
        text = {"no": desc.get("false") or desc.get("no"), "yes": desc.get("true") or desc.get("yes")}
        choices = [f"{label}: {text[label]}" if text[label] else label for label in NOUL_LABELS]
        return Question(question=instructions, choices=choices), {"type": "noul", "labels": list(NOUL_LABELS)}
    if qtype == "score":
        if not isinstance(criteria, list) or len(criteria) < 2:
            raise SystemOneError(f"question {name!r}: score needs a criteria list of >= 2 levels")
        labels = [str(i) for i in range(len(criteria))]
        choices = [f"{i}: {c}" if c else str(i) for i, c in enumerate(criteria)]
        return Question(question=instructions, choices=choices), {"type": "score", "labels": labels}
    raise SystemOneError(f"question {name!r}: unknown type {qtype!r}")


def format_answer(meta: dict, probs: list[float]) -> dict:
    labels = meta["labels"]
    dist = {label: float(p) for label, p in zip(labels, probs)}
    if meta["type"] == "noul":
        return {"type": "noul", "noul": dist["yes"]}
    if meta["type"] == "choice":
        best = max(dist, key=dist.get)
        return {"type": "choice", "choice": best, "probabilities": dist}
    expected = sum(i * p for i, p in enumerate(probs))
    return {"type": "score", "probabilities": dist, "score": expected}


def decide_systemone(engine, body: dict, model_name: str) -> dict:
    state = body.get("state")
    if not isinstance(state, str):
        raise SystemOneError("state must be a string")
    questions = body.get("questions")
    if not isinstance(questions, dict) or not questions:
        raise SystemOneError("questions must be a non-empty object")
    names, qs, metas = [], [], []
    for name, q in questions.items():
        question, meta = build_question(name, q)
        names.append(name)
        qs.append(question)
        metas.append(meta)
    decisions = engine.decide(state, qs)
    prefix = engine.rt.prompt.prefix_ids(state)
    n_in = len(prefix) + sum(len(engine.rt.prompt.suffix_ids(q.question, q.choices)) for q in qs)
    answers = {name: format_answer(meta, d.probabilities) for name, meta, d in zip(names, metas, decisions)}
    return {"answers": answers, "usage": {"input_tokens": n_in, "output_tokens": 0},
            "model": body.get("model") or model_name}
