"""Few-shot exemplars placed in the shared state (prefix).

With shared-prefix engines (packed / shared) the exemplars are prefilled once per state and cost nothing per
question, so k-shot prompting is almost free here, unlike per-question k-shot in ordinary generation.
Exemplars come from MMLU `dev` (5 per subject). JMMLU shares MMLU's subject names for most subjects; subjects
without dev exemplars fall back to the fixed cross-subject set.
"""

from __future__ import annotations

from functools import lru_cache

from jqv.prompt import LETTERS

HEADER = "Examples of questions answered correctly in the same format:"


@lru_cache(maxsize=1)
def dev_by_subject() -> dict[str, list[dict]]:
    from datasets import load_dataset

    out: dict[str, list[dict]] = {}
    for r in load_dataset("cais/mmlu", "all", split="dev"):
        out.setdefault(r["subject"], []).append(
            {"question": r["question"], "choices": list(r["choices"]), "answer": int(r["answer"])})
    return out


def format_exemplar(item: dict) -> str:
    opts = "\n".join(f"{LETTERS[i]}. {c}" for i, c in enumerate(item["choices"]))
    return f"Question:\n{item['question']}\n\nOptions:\n{opts}\n\nAnswer: {LETTERS[item['answer']]}"


def fewshot_state(subject: str | None, k: int = 5) -> str:
    """k exemplars of the same subject (falls back to the fixed cross-subject set)."""
    ex = dev_by_subject().get(subject or "", [])[:k]
    if len(ex) < k:
        return fixed_state(k)
    return HEADER + "\n\n" + "\n\n".join(format_exemplar(e) for e in ex)


@lru_cache(maxsize=8)
def fixed_state(k: int = 5) -> str:
    """One shared state for every question: k exemplars from k evenly spaced subjects (deterministic)."""
    subjects = sorted(dev_by_subject())
    picks = [subjects[int(i * len(subjects) / k)] for i in range(k)]
    return HEADER + "\n\n" + "\n\n".join(format_exemplar(dev_by_subject()[s][0]) for s in picks)
