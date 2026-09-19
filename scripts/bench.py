"""Throughput benchmark: state length x number of questions x engine.

    uv run scripts/bench.py --state-tokens 500 2000 8000 --questions 1 10 100 --engines generate naive kvcache packed
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import time
from pathlib import Path

import torch

from _common import RESULTS, add_model_args, dump_json, load_rt, slug
from jqv.engine import make_engine
from jqv.types import Question

FILLER = (
    "橋梁の定期点検では、主桁、横桁、床版、支承、伸縮装置、橋台、橋脚などの部材ごとに損傷の種類と程度を記録する。"
    "鋼部材では腐食、亀裂、ゆるみ・脱落、破断、防食機能の劣化を、コンクリート部材ではひび割れ、剥離・鉄筋露出、"
    "漏水・遊離石灰、抜け落ち、うきを主な損傷として評価する。"
)
QUESTIONS = [
    ("主桁に腐食は確認されているか。", ["確認されている", "確認されていない"]),
    ("床版の損傷として記述されているものはどれか。", ["ひび割れ", "うき", "漏水", "記述なし"]),
    ("緊急の措置が必要か。", ["必要", "不要", "判断できない"]),
    ("支承の状態はどれか。", ["健全", "軽微な損傷", "要補修", "記述なし"]),
]


def make_state(rt, n_tokens: int) -> str:
    ids = rt.tokenizer.encode(FILLER, add_special_tokens=False)
    reps = n_tokens // len(ids) + 1
    text = FILLER * reps
    ids = rt.tokenizer.encode(text, add_special_tokens=False)[:n_tokens]
    return rt.tokenizer.decode(ids)


def make_questions(n: int) -> list[Question]:
    return [Question(question=f"({i + 1}) {QUESTIONS[i % len(QUESTIONS)][0]}", choices=QUESTIONS[i % len(QUESTIONS)][1])
            for i in range(n)]


def driver_mem_gb(rt) -> float | None:
    """MPS driver-allocated memory (a high-water proxy: the allocator keeps freed blocks)."""
    return torch.mps.driver_allocated_memory() / 1e9 if rt.device.type == "mps" else None


def time_call(rt, fn, repeat: int, label: str, long_run_threshold: float = 60.0) -> tuple[float, float | None]:
    t = time.perf_counter()
    fn()  # warmup
    rt.sync()
    warm = time.perf_counter() - t
    mem = driver_mem_gb(rt)
    print(f"    {label} warmup {warm:6.1f}s", flush=True)
    if warm > long_run_threshold and repeat > 1:
        print(f"    {label} warmup exceeded {long_run_threshold:.0f}s -> timing 1 run instead of {repeat}", flush=True)
        repeat = 1
    ts = []
    for i in range(repeat):
        t = time.perf_counter()
        fn()
        rt.sync()
        ts.append(time.perf_counter() - t)
        print(f"    {label} run {i + 1}/{repeat} {ts[-1]:6.1f}s", flush=True)
    return statistics.median(ts), mem


def cost_tokens(engine: str, s_len: int, nq: int, q_tok: int) -> int:
    """Tokens the engine actually pushes through the model (used only for ETA estimates)."""
    if engine in ("generate", "naive"):
        return nq * s_len + q_tok
    return s_len + q_tok


def eta_seconds(remaining, tps: dict, repeat: int, default_tps: float) -> float:
    total = 0.0
    for s_len, nq, q_tok, label in remaining:
        name = label.split(":")[0]
        rate = tps.get(name) or tps.get("naive") or default_tps
        total += cost_tokens(name, s_len, nq, q_tok) / rate * (repeat + 1)
    return total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--state-tokens", type=int, nargs="+", default=[500, 2000, 8000])
    ap.add_argument("--questions", type=int, nargs="+", default=[1, 10, 100])
    ap.add_argument("--engines", nargs="+", default=["generate", "naive", "kvcache", "packed"])
    ap.add_argument("--repeat", type=int, default=3)
    ap.add_argument("--max-generate-questions", type=int, default=100)
    ap.add_argument("--readout", default="full", choices=["full", "rows"],
                    help="full = B (full LM head), rows = B' (letters' rows only); rows are labelled engine:rows")
    ap.add_argument("--out", default=None,
                    help="incremental JSONL, one row per finished condition (default results/bench_<model>.jsonl). "
                         "Existing rows are reused, so a killed run resumes where it stopped.")
    ap.add_argument("--long-run-threshold", type=float, default=60.0,
                    help="if the warmup run takes longer than this (s), time only 1 run for that condition")
    add_model_args(ap)
    a = ap.parse_args()

    rt = load_rt(a)
    plan = []
    for s_tok in a.state_tokens:
        state = make_state(rt, s_tok)
        s_len = len(rt.prompt.prefix_ids(state))
        for nq in a.questions:
            qs = make_questions(nq)
            q_tok = sum(len(rt.prompt.suffix_ids(q.question, q.choices)) for q in qs)
            for name in a.engines:
                if name == "generate" and nq > a.max_generate_questions:
                    continue
                label = name if (a.readout == "full" or name == "generate") else f"{name}:{a.readout}"
                plan.append((state, qs, s_len, nq, q_tok, label))
    out_jsonl = Path(a.out) if a.out else RESULTS / f"bench_{slug(rt.model_id)}.jsonl"
    done: dict[tuple, dict] = {}
    if out_jsonl.exists():
        for line in out_jsonl.read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                done[(r["engine"], r["state_tokens"], r["questions"])] = r
        print(f"resume: {len(done)} finished conditions found in {out_jsonl}", flush=True)
    default_tps = 4000.0
    tps: dict[str, float] = {}
    remaining = [(p[2], p[3], p[4], p[5]) for p in plan]
    print(f"{len(plan)} conditions, repeat={a.repeat} (+1 warmup each). "
          f"initial rough ETA ~{eta_seconds(remaining, tps, a.repeat, default_tps) / 60:.0f} min "
          f"(assumes {default_tps:.0f} tok/s; refined after each condition)", flush=True)
    rows = []
    t_start = time.time()
    for idx, (state, qs, s_len, nq, q_tok, label) in enumerate(plan, 1):
        name = label.split(":")[0]
        if (label, s_len, nq) in done:
            row = done[(label, s_len, nq)]
            rows.append(row)
            tps[name] = cost_tokens(name, s_len, nq, q_tok) / row["seconds"]
            print(f"[{idx}/{len(plan)}] {label} S={s_len} Q={nq} already done ({row['seconds'] * 1000:.0f} ms), skipped", flush=True)
            continue
        print(f"[{idx}/{len(plan)}] {label} S={s_len} Q={nq} ({cost_tokens(name, s_len, nq, q_tok)} tok/run)", flush=True)
        eng = make_engine(name, rt, readout=a.readout)
        sec, mem = time_call(rt, lambda: eng.decide(state, qs), a.repeat, label, a.long_run_threshold)
        tps[name] = cost_tokens(name, s_len, nq, q_tok) / sec
        row = {"engine": label, "readout": "full" if name == "generate" else a.readout,
               "state_tokens": s_len, "questions": nq, "question_tokens": q_tok,
               "seconds": sec, "questions_per_sec": nq / sec,
               "naive_equiv_tokens_per_sec": (nq * s_len + q_tok) / sec, "driver_mem_gb": mem}
        rows.append(row)
        append_row(out_jsonl, row)  # saved immediately; a killed run loses at most the current condition
        done[(label, s_len, nq)] = row
        write_summary(list(done.values()), rt)
        remaining = [(p[2], p[3], p[4], p[5]) for p in plan[idx:]]
        eta = eta_seconds(remaining, tps, a.repeat, default_tps)
        print(f"{label:<12} S={s_len:>6} Q={nq:>4}  {sec * 1000:9.1f} ms  {nq / sec:8.1f} q/s   "
              f"| elapsed {(time.time() - t_start) / 60:.1f} min, ETA remaining ~{eta / 60:.1f} min", flush=True)
    write_summary(list(done.values()), rt)


ENGINE_ORDER = {"generate": 0, "naive": 1, "kvcache": 2, "packed": 3, "shared": 4}


def write_summary(rows, rt) -> None:
    """Regenerate the JSON + Markdown summary from all saved rows (called after every condition)."""
    rows = sorted(rows, key=lambda r: (r["state_tokens"], r["questions"], ENGINE_ORDER.get(r["engine"].split(":")[0], 9), r["engine"]))
    base = {(r["state_tokens"], r["questions"]): r["seconds"] for r in rows if r["engine"] == "naive"}
    for r in rows:
        b = base.get((r["state_tokens"], r["questions"]))
        r["speedup_vs_naive"] = (b / r["seconds"]) if b else None
    out = RESULTS / f"bench_{slug(rt.model_id)}.json"
    out.write_text(json.dumps({"model": rt.model_id, "dtype": str(rt.dtype), "device": str(rt.device), "rows": rows},
                              indent=2, ensure_ascii=False))
    md = ["| engine | state tok | Q | ms | q/s | speedup vs naive | driver mem GB |", "|---|---:|---:|---:|---:|---:|---:|"]
    for r in rows:
        sp = f"{r['speedup_vs_naive']:.2f}x" if r["speedup_vs_naive"] else "-"
        mem = f"{r['driver_mem_gb']:.1f}" if r.get("driver_mem_gb") else "-"
        md.append(f"| {r['engine']} | {r['state_tokens']} | {r['questions']} | {r['seconds'] * 1000:.0f} | {r['questions_per_sec']:.1f} | {sp} | {mem} |")
    (RESULTS / f"bench_{slug(rt.model_id)}.md").write_text("\n".join(md) + "\n")


def append_row(path: Path, row: dict) -> None:
    with path.open("a") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
        f.flush()
        os.fsync(f.fileno())


if __name__ == "__main__":
    main()
