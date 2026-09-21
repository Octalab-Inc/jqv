"""The cascade: walk -> lexical sketch (stage 0) -> jqv file routing (stage 1) -> full-file shared-state range scoring (stage 2)."""

from __future__ import annotations

import sys
import time
from dataclasses import dataclass

from jqgrep.candidate import stage1
from jqgrep.output import Hit
from jqgrep.passages import ROLE_LABELS, Range, build_state, extra_questions, line_ranges, merge_ranges, range_question, trim_range, windows
from jqgrep.sketch import Sketch, stage0
from jqgrep.walker import walk


@dataclass
class SearchConfig:
    n0: int = 150            # lexical candidates kept for stage 1
    n1: int = 20             # files kept after stage 1
    gap1: float = 0.25       # stage 1 relative threshold: p >= best - gap1
    k: int = 10              # final hits returned
    gap2: float = 0.25       # stage 2 relative threshold on range probabilities
    threshold: float | None = None  # optional absolute cut on range probability (not calibrated for code; off by default)
    range_size: int = 40
    stride: int = 30
    max_ranges: int = 24           # <= 26 so that one listwise question can cover every range
    max_state_tokens: int = 7000
    stage1_batch: int = 40
    snippet_lines: int = 3
    max_file_bytes: int = 1_000_000
    skip_stage1: bool = False
    code_only: bool = True         # discount docs / data files in the lexical stage
    group_size: int = 12           # sketches per shared state in stage 1
    trace: bool = False            # log the full stage-1 ranking


def _log(msg: str):
    print(msg, file=sys.stderr, flush=True)


def score_file(client, query: str, sk: Sketch, cfg: SearchConfig) -> tuple[list[tuple[Range, float]], dict]:
    """Stage 2 for one file: one decide() per window (usually one) with all range questions plus the file-level questions."""
    lines = sk.text.splitlines()
    n = len(lines)
    total_tokens = client.n_tokens(sk.text)
    per_line = max(1.0, total_tokens / max(1, n))
    max_lines = max(60, int(cfg.max_state_tokens / per_line))
    scored: list[tuple[Range, float]] = []
    meta = {"file_relevant": 0.0, "role": ROLE_LABELS[0], "role_p": 0.0, "production": 0.0, "ambiguous": 0.0, "windows": 0, "ranges": 0}
    for w in windows(n, max_lines):
        wl = lines[w.start - 1 : w.end]
        ranges = [Range(r.start + w.start - 1, r.end + w.start - 1) for r in line_ranges(len(wl), cfg.range_size, cfg.stride, cfg.max_ranges)]
        state = build_state(query, sk.entry.rel, sk.entry.language, wl, w.start)
        qs = [(range_question(r), ["yes", "no"]) for r in ranges] + [(q, c) for _, q, c in extra_questions()]
        listwise = 2 <= len(ranges) <= 26
        if listwise:
            qs.append(("Which of these line ranges is the most relevant to the search query?", [f"lines {r.start}-{r.end}" for r in ranges]))
        probs = client.decide(state, qs)
        lw = probs[-1] if listwise else [1.0] * len(ranges)
        top = max(lw) or 1.0
        for r, p, l in zip(ranges, probs[: len(ranges)], lw):
            scored.append((r, p[0] * (0.5 + 0.5 * l / top)))
        extras = probs[len(ranges) : len(ranges) + len(extra_questions())]
        keys = [k for k, _, _ in extra_questions()]
        ex = dict(zip(keys, extras))
        # keep the window with the highest file-level relevance as the file's verdict
        if ex["file_relevant"][0] >= meta["file_relevant"]:
            role_i = max(range(len(ROLE_LABELS)), key=lambda i: ex["role"][i])
            meta.update(file_relevant=ex["file_relevant"][0], role=ROLE_LABELS[role_i], role_p=ex["role"][role_i],
                        production=ex["production"][0], ambiguous=ex["ambiguous"][0])
        meta["windows"] += 1
        meta["ranges"] += len(ranges)
    return scored, meta


def search(query: str, root: str, cfg: SearchConfig, client, log=_log) -> tuple[list[Hit], dict]:
    t0 = time.time()
    entries = walk(root, max_bytes=cfg.max_file_bytes)
    log(f"walked {len(entries)} files under {root}")
    sketches = stage0(entries, query, cfg.n0, cfg.code_only)
    log(f"stage 0: {len(sketches)} lexical candidates (top: {', '.join(s.entry.rel for s in sketches[:5])})")
    stage1_p: dict[str, float] = {}
    if cfg.skip_stage1:
        routed = sketches[: cfg.n1]
    else:
        kept, scored = stage1(client, query, sketches, cfg.n1, cfg.gap1, cfg.group_size, log)
        stage1_p = {s.entry.rel: p for s, p in scored}
        routed = [s for s, _ in kept]
        log("stage 1: kept " + ", ".join(f"{s.entry.rel} ({p:.2f})" for s, p in kept[:8]) + (" ..." if len(kept) > 8 else ""))
        if cfg.trace:
            for rank, (s, p) in enumerate(scored[:30], 1):
                log(f"    #{rank:2d} {p:.3f} lexical={s.lexical:.1f} {s.entry.rel}")
    hits: list[Hit] = []
    n_ranges = 0
    for i, sk in enumerate(routed):
        scored, meta = score_file(client, query, sk, cfg)
        n_ranges += meta["ranges"]
        if not scored:
            continue
        best = max(p for _, p in scored)
        min_p = best - cfg.gap2 if cfg.threshold is None else max(cfg.threshold, best - cfg.gap2)
        lines = sk.text.splitlines()
        for r, p in merge_ranges(scored, min_p):
            tr = trim_range(lines, r)
            hits.append(Hit(path=sk.entry.rel, start=tr.start, end=tr.end, score=round(p, 4), file_score=round(meta["file_relevant"], 4),
                            role=meta["role"], role_p=round(meta["role_p"], 4), production=round(meta["production"], 4),
                            ambiguous=round(meta["ambiguous"], 4), snippet=[ln for ln in lines[tr.start - 1 : tr.end] if ln.strip()][: cfg.snippet_lines],
                            stage1=round(stage1_p[sk.entry.rel], 4) if sk.entry.rel in stage1_p else None))
        log(f"  stage 2: [{i + 1}/{len(routed)}] {sk.entry.rel}: best range p {best:.2f}, file {meta['file_relevant']:.2f}, role {meta['role']}")
    hits.sort(key=lambda h: (-(h.score * (0.6 + 0.4 * h.file_score)), h.path, h.start))
    if hits:
        top = hits[0].score
        hits = [h for h in hits if h.score >= top - cfg.gap2 and (cfg.threshold is None or h.score >= cfg.threshold)][: cfg.k]
    stats = {"walked": len(entries), "stage0": len(sketches), "stage1": len(routed), "stage2_files": len(routed), "ranges": n_ranges,
             "seconds": round(time.time() - t0, 1), "model": getattr(client, "model_id", "?"), "engine": getattr(client, "engine_name", "?"),
             "temperature": getattr(client, "temperature", None)}
    return hits, stats
