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
        if run.endswith("_permavg"):  # perm_avg variants are reported in their own column
            continue
        out[run] = {"metrics": jload(f), "calibration": jload(f[:-5] + "_calibration.json")}
    return out


def variant(slug: str, ds: str, tag: str):
    m = jload(RESULTS / f"{ds}_packed_{slug}{tag}.json")
    c = jload(RESULTS / f"{ds}_packed_{slug}{tag}_calibration.json")
    return m, c


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--slot-run", nargs="*", default=[], help="slug=run to pin the slot+LoRA run used per model")
    a = ap.parse_args()
    pinned = dict(s.split("=", 1) for s in a.slot_run)
    # base model slugs only (qwen3-1.7b, qwen3-14b, ...); variant caches carry a _suffix and are read as columns
    slugs = sorted({re.match(r"mmlu_packed_(qwen3-[0-9.]+b)\.json$", f.split("/")[-1]).group(1)
                    for f in glob.glob(str(RESULTS / "mmlu_packed_qwen3-*.json"))
                    if re.match(r"mmlu_packed_(qwen3-[0-9.]+b)\.json$", f.split("/")[-1])},
                   key=lambda s: float(re.search(r"([\d.]+)b", s).group(1)))
    rows = []
    for slug in slugs:
        raw = {ds: variant(slug, ds, "")[0] for ds in ("mmlu", "jmmlu")}
        cal = {ds: variant(slug, ds, "")[1] for ds in ("mmlu", "jmmlu")}
        pa = {ds: variant(slug, ds, "_permavg") for ds in ("mmlu", "jmmlu")}
        fs = variant(slug, "mmlu", "_shots5_permavg")
        runs = slot_runs(slug, "mmlu")
        run = pinned.get(slug)
        if run is None and runs:  # default: lowest MMLU NLL after temperature (val-fitted)
            run = min(runs, key=lambda r: (runs[r]["calibration"] or {}).get("after", {}).get("nll", 9e9))
        slot = runs.get(run) if run else None
        slot_j = slot_runs(slug, "jmmlu").get(run) if run else None
        slot_pa = jload(RESULTS / f"mmlu_slot_{slug}_{run}_permavg.json") if run else None
        qps = qps_pa = None
        bench = RESULTS / f"bench_{slug}.jsonl"
        if bench.exists():
            for line in bench.read_text().splitlines():
                r = json.loads(line)
                if r["questions"] == 100 and 2000 <= r["state_tokens"] <= 2100:
                    if r["engine"] == "packed":
                        qps = r["questions_per_sec"]
                    elif r["engine"] == "packed:permavg":
                        qps_pa = r["questions_per_sec"]
        g = lambda m, k: (m[k] if m else None)
        jb = {}
        jb_path = RESULTS / "jevbench_summary.json"
        if jb_path.exists():
            for r in json.loads(jb_path.read_text())["rows"]:
                if r["label"].startswith(slug + "_"):
                    jb[r["label"].split("_", 1)[1]] = r["tiers"].get("hard", {}).get("accuracy")
        rows.append({
            "model": slug, "params": PARAMS.get(slug, "?"),
            "jevbench_hard_packed": jb.get("packed"), "jevbench_hard_permavg": jb.get("permavg"), "jevbench_hard_slot": jb.get("slot"),
            "mmlu_acc_raw": g(raw["mmlu"], "accuracy"), "jmmlu_acc_raw": g(raw["jmmlu"], "accuracy"),
            "ece_raw": cal["mmlu"]["before"]["ece"] if cal["mmlu"] else g(raw["mmlu"], "ece"),
            "ece_T": cal["mmlu"]["after"]["ece"] if cal["mmlu"] else None,
            "T": cal["mmlu"]["temperature"] if cal["mmlu"] else None,
            "mmlu_acc_permavg": g(pa["mmlu"][0], "accuracy"), "jmmlu_acc_permavg": g(pa["jmmlu"][0], "accuracy"),
            "permavg_ece_raw": pa["mmlu"][1]["before"]["ece"] if pa["mmlu"][1] else g(pa["mmlu"][0], "ece"),
            "permavg_ece_T": pa["mmlu"][1]["after"]["ece"] if pa["mmlu"][1] else None,
            "mmlu_acc_shots5_permavg": g(fs[0], "accuracy"),
            "slot_run": run,
            "mmlu_acc_slot": slot["metrics"]["accuracy"] if slot else None,
            "jmmlu_acc_slot": slot_j["metrics"]["accuracy"] if slot_j else None,
            "slot_ece_raw": slot["calibration"]["before"]["ece"] if slot and slot["calibration"] else (slot["metrics"]["ece"] if slot else None),
            "slot_ece_T": slot["calibration"]["after"]["ece"] if slot and slot["calibration"] else None,
            "mmlu_acc_slot_permavg": slot_pa["accuracy"] if slot_pa else None,
            "packed_qps_s2k_q100": qps, "packed_permavg_qps_s2k_q100": qps_pa,
        })
    f = lambda x, d=3: "-" if x is None else (f"{x:.{d}f}" if isinstance(x, float) else str(x))
    md = ["| backbone | params | B zero-shot: MMLU / JMMLU | B ECE raw / +T (T) | B + perm_avg: MMLU / JMMLU | perm_avg ECE raw / +T | 5-shot + perm_avg: MMLU | slot+LoRA: MMLU / JMMLU | slot ECE raw / +T | slot + perm_avg: MMLU | JevBench hard (public 111): B / perm_avg / slot | packed q/s (S=2k, Q=100) plain / perm_avg |",
          "|---|---:|---:|---|---:|---|---:|---:|---|---:|---:|---:|"]
    for r in rows:
        md.append(f"| {r['model']} | {r['params']} | {f(r['mmlu_acc_raw'])} / {f(r['jmmlu_acc_raw'])} | "
                  f"{f(r['ece_raw'])} / {f(r['ece_T'])} ({f(r['T'], 1)}) | {f(r['mmlu_acc_permavg'])} / {f(r['jmmlu_acc_permavg'])} | "
                  f"{f(r['permavg_ece_raw'])} / {f(r['permavg_ece_T'])} | {f(r['mmlu_acc_shots5_permavg'])} | "
                  f"{f(r['mmlu_acc_slot'])} / {f(r['jmmlu_acc_slot'])}{(' (' + r['slot_run'] + ')') if r['slot_run'] else ''} | "
                  f"{f(r['slot_ece_raw'])} / {f(r['slot_ece_T'])} | {f(r['mmlu_acc_slot_permavg'])} | "
                  f"{f(r['jevbench_hard_packed'])} / {f(r['jevbench_hard_permavg'])} / {f(r['jevbench_hard_slot'])} | "
                  f"{f(r['packed_qps_s2k_q100'], 1)} / {f(r['packed_permavg_qps_s2k_q100'], 1)} |")
    md.append(f"| {JEV['model']} | ? | **{JEV['mmlu_acc_raw']:.3f}** / - | {JEV['ece_raw']:.3f} (zero-shot, no T) | - | - | - | - | - | - | **0.741** (534 決定、Benchmark Heaven) | 30k tok in ~160 ms |")
    print("\n".join(md))
    dump_json({"rows": rows, "jev": JEV}, RESULTS / "scaling_table.json")
    (RESULTS / "scaling_table.md").write_text("\n".join(md) + "\n")


if __name__ == "__main__":
    main()
