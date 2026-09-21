"""Stage 1: semantic routing over sketches, jqv-style.

The sketches of a group of files go into ONE shared state; the questions are short: one yes/no per file plus one
listwise "which file" question over the same files. The shared state is prefilled once per group, the per-file
branches are a few tokens each, and the listwise question makes the model compare files instead of saying "yes" to
each in isolation. Two passes: all candidates in groups, then a final group of the best ones."""

from __future__ import annotations

from jqgrep.sketch import Sketch

MAX_LISTWISE = 26  # one option letter per file


def _group_state(query: str, group: list[Sketch]) -> str:
    parts = [f"Search query: {query}", "",
             "Below are sketches of several source files (path, language, top-level symbols, imports, the start of the file and lines "
             "containing query words). The questions ask which of these files contain what the query asks for; prefer files that "
             "implement or define it over files that merely mention it.", ""]
    for i, s in enumerate(group, 1):
        parts.append(f"===== [{i}] {s.entry.rel} =====")
        parts.append(s.prose())
        parts.append("")
    return "\n".join(parts)


def score_group(client, query: str, group: list[Sketch]) -> list[float]:
    """Combined score per file in the group: p(yes) x (0.5 + 0.5 x listwise share relative to the best file)."""
    state = _group_state(query, group)
    qs = [(f"Is file [{i}] {s.entry.rel} likely to contain what the search query asks for?", ["yes", "no"]) for i, s in enumerate(group, 1)]
    listwise = len(group) <= MAX_LISTWISE and len(group) >= 2
    if listwise:
        qs.append(("Which of these files most likely contains what the search query asks for?", [f"[{i}] {s.entry.rel}" for i, s in enumerate(group, 1)]))
    probs = client.decide(state, qs)
    p_yes = [p[0] for p in probs[: len(group)]]
    if listwise:
        lw = probs[len(group)]
        top = max(lw) or 1.0
        return [py * (0.5 + 0.5 * l / top) for py, l in zip(p_yes, lw)]
    return p_yes


def stage1(client, query: str, sketches: list[Sketch], keep: int, gap: float, group_size: int = 12, log=None) -> tuple[list[tuple[Sketch, float]], list[tuple[Sketch, float]]]:
    """Returns (kept, all_scored). kept = top `keep` files whose score is within `gap` of the best one (after the final pass)."""
    if not sketches:
        return [], []
    scored: dict[str, tuple[Sketch, float]] = {}
    groups = [sketches[i : i + group_size] for i in range(0, len(sketches), group_size)]
    for gi, group in enumerate(groups):
        for s, sc in zip(group, score_group(client, query, group)):
            scored[s.entry.rel] = (s, sc)
        if log:
            log(f"  stage 1: group {gi + 1}/{len(groups)} scored ({len(group)} files)")
    ranked = sorted(scored.values(), key=lambda x: (-x[1], x[0].entry.rel))
    # final pass: the best 2*keep files compete directly in groups of <= MAX_LISTWISE
    finalists = [s for s, _ in ranked[: min(len(ranked), max(keep * 2, 4))]]
    if len(finalists) > 2:
        final_scores: dict[str, float] = {}
        for i in range(0, len(finalists), min(group_size * 2, MAX_LISTWISE)):
            group = finalists[i : i + min(group_size * 2, MAX_LISTWISE)]
            for s, sc in zip(group, score_group(client, query, group)):
                final_scores[s.entry.rel] = sc
        if log:
            log(f"  stage 1: final pass over {len(finalists)} files")
        for rel, sc in final_scores.items():
            s, first = scored[rel]
            scored[rel] = (s, 0.5 * first + 0.5 * sc)
    ranked = sorted(scored.values(), key=lambda x: (-x[1], x[0].entry.rel))
    best = ranked[0][1]
    kept = [(s, p) for s, p in ranked[:keep] if p >= best - gap]
    return kept, ranked
