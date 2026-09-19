"""Backbone scaling table: raw vocab readout (B) vs slot+LoRA, calibration and throughput per model, plus Jev.

    uv run scripts/scaling_table.py                      # all models found in results/
    uv run scripts/scaling_table.py --slot-run qwen3-14b=qwen3-14b_slot_brier0   # pin the slot run per model

Reads results/{mmlu,jmmlu}_packed_<slug>.json (+ _calibration.json), results/mmlu_slot_<slug>_<run>*.json,
results/bench_<slug>.jsonl (packed, S=2038, Q=100). Jev numbers are Hume's measurements (MMLU 1,200 items, 10-bin ECE).
"""

from __future__ import annotations

import argparse
import glob
import json
import re

from _common import RESULTS, dump_json

PARAMS = {"qwen3-1.7b": "1.7B", "qwen3-4b": "4B", "qwen3-8b": "8B", "qwen3-14b": "14.8B", "qwen3-32b": "32.8B"}
JEV = {"model": "Jev (TypeSafe, Hume 2025)", "params": "?", "mmlu_acc_raw": 0.918, "ece_raw": 0.031, "note": "zero-shot; no temperature"}


def jload(path):
    try:
        return json.loads(open(path).read())
    except FileNotFoundError:
        return None


def slot_runs(slug: str, ds: str) -> dict[str, dict]:
    out = {}
    for f in glob.glob(str(RESULTS / f"{ds}_slot_{slug}_*.json")):
        name = re.sub(r"_calibration$|_temperature$", "", f.split("/")[-1][:-5])
        if name.endswith("_temperature") or "_calibration" in f or "_temperature" in f:
            continue
        run = name.split(f"{ds}_slot_{slug}_", 1)[1]
        out[run] = {"metrics": jload(f), "calibration": jload(f[:-5] + "_calibration.json")}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--slot-run", nargs="*", default=[], help="slug=run to pin the slot+LoRA run used per model")
    a = ap.parse_args()
    pinned = dict(s.split("=", 1) for s in a.slot_run)
    slugs = sorted({re.search(r"mmlu_packed_(qwen3-.+?)\.json$", f.split("/")[-1]).group(1)
                    for f in glob.glob(str(RESULTS / "mmlu_packed_qwen3-*.json"))
                    if "_calibration" not in f and "_temperature" not in f},
                   key=lambda s: float(re.search(r"([\d.]+)b", s).group(1)))
    rows = []
    for slug in slugs:
        raw = {ds: jload(RESULTS / f"{ds}_packed_{slug}.json") for ds in ("mmlu", "jmmlu")}
        cal = {ds: jload(RESULTS / f"{ds}_packed_{slug}_calibration.json") for ds in ("mmlu", "jmmlu")}
        runs = slot_runs(slug, "mmlu")
        run = pinned.get(slug)
        if run is None and runs:  # default: lowest MMLU NLL after temperature (val-fitted)
            run = min(runs, key=lambda r: (runs[r]["calibration"] or {}).get("after", {}).get("nll", 9e9))
        slot = runs.get(run) if run else None
        slot_j = slot_runs(slug, "jmmlu").get(run) if run else None
        qps = None
        bench = RESULTS / f"bench_{slug}.jsonl"
        if bench.exists():
            for line in bench.read_text().splitlines():
                r = json.loads(line)
                if r["engine"] == "packed" and r["questions"] == 100 and 2000 <= r["state_tokens"] <= 2100:
                    qps = r["questions_per_sec"]
        rows.append({
            "model": slug, "params": PARAMS.get(slug, "?"),
            "mmlu_acc_raw": raw["mmlu"]["accuracy"] if raw["mmlu"] else None,
            "jmmlu_acc_raw": raw["jmmlu"]["accuracy"] if raw["jmmlu"] else None,
            "ece_raw": cal["mmlu"]["before"]["ece"] if cal["mmlu"] else (raw["mmlu"]["ece"] if raw["mmlu"] else None),
            "ece_T": cal["mmlu"]["after"]["ece"] if cal["mmlu"] else None,
            "T": cal["mmlu"]["temperature"] if cal["mmlu"] else None,
            "slot_run": run,
            "mmlu_acc_slot": slot["metrics"]["accuracy"] if slot else None,
            "jmmlu_acc_slot": slot_j["metrics"]["accuracy"] if slot_j else None,
            "slot_ece_raw": slot["calibration"]["before"]["ece"] if slot and slot["calibration"] else (slot["metrics"]["ece"] if slot else None),
            "slot_ece_T": slot["calibration"]["after"]["ece"] if slot and slot["calibration"] else None,
            "packed_qps_s2k_q100": qps,
        })
    f = lambda x, d=3: "-" if x is None else (f"{x:.{d}f}" if isinstance(x, float) else str(x))
    md = ["| backbone | params | B: MMLU acc | B: JMMLU acc | B: ECE raw / +T (T) | slot+LoRA run | slot: MMLU acc | slot: JMMLU acc | slot: ECE raw / +T | packed q/s (S=2k, Q=100) |",
          "|---|---:|---:|---:|---|---|---:|---:|---|---:|"]
    for r in rows:
        md.append(f"| {r['model']} | {r['params']} | {f(r['mmlu_acc_raw'])} | {f(r['jmmlu_acc_raw'])} | "
                  f"{f(r['ece_raw'])} / {f(r['ece_T'])} ({f(r['T'], 1)}) | {r['slot_run'] or '-'} | {f(r['mmlu_acc_slot'])} | "
                  f"{f(r['jmmlu_acc_slot'])} | {f(r['slot_ece_raw'])} / {f(r['slot_ece_T'])} | {f(r['packed_qps_s2k_q100'], 1)} |")
    md.append(f"| {JEV['model']} | ? | **{JEV['mmlu_acc_raw']:.3f}** | - | {JEV['ece_raw']:.3f} (zero-shot, no T) | - | - | - | - | 30k tok in ~160 ms |")
    print("\n".join(md))
    dump_json({"rows": rows, "jev": JEV}, RESULTS / "scaling_table.json")
    (RESULTS / "scaling_table.md").write_text("\n".join(md) + "\n")


if __name__ == "__main__":
    main()
