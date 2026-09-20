"""Aggregate results/jevbench/<label>/<tier>/summary.json into one table with the JevBench axes we can compute
locally (public tiers only: easy 48, standard 72, hard 111; the judge tier and the 20 distribution-gold items are
held out, so Intelligence is renormalized over the available tiers and Calibration uses hard-tier ECE only).

    uv run scripts/jevbench_summary.py
"""

from __future__ import annotations

import glob
import json
import math
import os
import subprocess
import sys
from pathlib import Path

from _common import RESULTS, dump_json

JEVBENCH = Path(os.environ.get("JEVBENCH_DIR", "/Users/h.imura/tmp/repo/jevbench"))
TIER_FILES = {"easy": "easy.jsonl", "standard": "original.jsonl", "hard": "hard.jsonl"}

TIER_WEIGHTS = {"easy": 0.14, "standard": 0.28, "judge": 0.28, "hard": 0.30}  # composite_v12.py
LEADERBOARD = [  # benchmarkheaven.com/jev-models, v1.2.5 (2026-09-20); Intelligence includes judge + held-out items
    {"label": "Jev 1.13 (TypeSafe)", "intelligence": 90.4, "calibration": 82.7, "hard": 0.741, "p50": 0.65, "score": 75.4},
    {"label": "SemIf Qwen3.5-4B", "intelligence": 85.9, "calibration": 72.6, "hard": 0.595, "p50": 0.20, "score": 74.7},
    {"label": "openjev-sglang Qwen3.6-35B-A3B", "intelligence": 88.9, "calibration": None, "hard": 0.714, "p50": 0.68, "score": None},
    {"label": "GPT-5.6 Luna (low)", "intelligence": 96.8, "calibration": 89.8, "hard": 0.945, "p50": 0.97, "score": 66.2},
]


def ece_value(x):
    if isinstance(x, dict):
        return x.get("ece", x.get("value"))
    return x


def temperature_for(cfg: dict) -> tuple[float | None, str | None]:
    """The MMLU-val temperature that the same engine configuration would ship with (provenance-checked file)."""
    slug = cfg["model"].split("/")[-1].lower()
    if cfg["engine"] in ("slot", "pointer") and cfg.get("head_dir"):
        run = Path(cfg["head_dir"]).parent.name
        cand = RESULTS / f"mmlu_{cfg['engine']}_{slug}_{run}{'_permavg' if cfg.get('perm_avg') else ''}_temperature.json"
    else:
        cand = RESULTS / f"mmlu_packed_{slug}{'_permavg' if cfg.get('perm_avg') else ''}_temperature.json"
    if cand.exists():
        return json.loads(cand.read_text())["temperature"], cand.name
    return None, None


def rescale(probs: dict, T: float) -> dict:
    """Temperature scaling on the returned distribution: p' ∝ p^(1/T) (identical to softmax(z/T))."""
    w = {k: max(float(v), 1e-12) ** (1.0 / T) for k, v in probs.items()}
    z = sum(w.values())
    return {k: v / z for k, v in w.items()}


def summarize_with_temperature(run_dir: Path, tier: str, T: float) -> dict | None:
    """Re-score a tier with temperature-scaled probabilities using the harness's own summarize (argmax and
    therefore accuracy are unchanged; ECE / Brier change). Writes <tier>/results_T.jsonl and summary_T.json."""
    src = run_dir / tier / "results.jsonl"
    if not src.exists():
        return None
    out = run_dir / tier / "results_T.jsonl"
    with out.open("w") as f:
        for line in src.read_text().splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            if r.get("ok") and isinstance(r.get("probs"), dict):
                r["probs"] = rescale(r["probs"], T)
                r["probs_as_returned"] = r["probs"]
            f.write(json.dumps(r) + "\n")
    res = subprocess.run([sys.executable, "-m", "jevbench.cli", "summarize", "--tasks",
                          str(JEVBENCH / "datasets" / "public" / TIER_FILES[tier]), "--results", str(out)],
                         cwd=JEVBENCH, capture_output=True, text=True)
    try:
        summ = json.loads(res.stdout)
    except json.JSONDecodeError:
        return None
    (run_dir / tier / "summary_T.json").write_text(res.stdout)
    return summ


def main():
    rows = []
    for cfg_path in sorted(glob.glob(str(RESULTS / "jevbench" / "*" / "config.json"))):
        run_dir = Path(cfg_path).parent
        cfg = json.loads(Path(cfg_path).read_text())
        tiers = {}
        for tier in ("easy", "standard", "hard"):
            f = run_dir / tier / "summary.json"
            if f.exists() and f.read_text().strip():
                s = json.loads(f.read_text())
                tiers[tier] = {"accuracy": s.get("accuracy"), "n": s.get("n_scorable"), "ece": ece_value(s.get("ece")),
                               "brier": s.get("brier_mean"), "latency": s.get("latency") or {},
                               "n_failed": (s.get("n_attempted") or 0) - (s.get("n_scorable") or 0)}
        if not tiers:
            continue
        num = sum(TIER_WEIGHTS[t] * v["accuracy"] for t, v in tiers.items() if v["accuracy"] is not None)
        den = sum(TIER_WEIGHTS[t] for t, v in tiers.items() if v["accuracy"] is not None)
        intel = 100 * num / den if den else None
        hard_ece = tiers.get("hard", {}).get("ece")
        calib = max(0.0, 100 * (1 - hard_ece / 0.5)) if hard_ece is not None else None
        lat = [v["latency"].get("p50_s") for v in tiers.values() if v["latency"].get("p50_s") is not None]
        T, t_file = temperature_for(cfg)
        hard_ece_T = calib_T = None
        if T is not None and "hard" in tiers:
            sT = summarize_with_temperature(run_dir, "hard", T)
            if sT:
                hard_ece_T = ece_value(sT.get("ece"))
                calib_T = max(0.0, 100 * (1 - hard_ece_T / 0.5)) if hard_ece_T is not None else None
        rows.append({"label": run_dir.name, **cfg, "tiers": tiers, "intelligence_public": intel,
                     "calibration_public": calib, "hard_ece": hard_ece, "p50_raw_s": (sum(lat) / len(lat)) if lat else None,
                     "temperature": T, "temperature_file": t_file, "hard_ece_T": hard_ece_T, "calibration_public_T": calib_T})
    md = ["| run | easy | standard | hard | Intelligence (public tiers) | hard ECE raw / +T (T) | Calibration raw / +T (ECE only) | raw p50 s |",
          "|---|---:|---:|---:|---:|---|---|---:|"]
    f = lambda x, d=3: "-" if x is None else f"{x:.{d}f}"
    for r in rows:
        t = r["tiers"]
        md.append(f"| {r['label']} | {f(t.get('easy', {}).get('accuracy'))} | {f(t.get('standard', {}).get('accuracy'))} | "
                  f"{f(t.get('hard', {}).get('accuracy'))} | {f(r['intelligence_public'], 1)} | "
                  f"{f(r['hard_ece'])} / {f(r['hard_ece_T'])} ({f(r['temperature'], 1)}) | "
                  f"{f(r['calibration_public'], 1)} / {f(r['calibration_public_T'], 1)} | {f(r['p50_raw_s'], 2)} |")
    for l in LEADERBOARD:
        md.append(f"| {l['label']} (Benchmark Heaven v1.2.5, 534 決定) | - | - | {f(l['hard'])} | {f(l['intelligence'], 1)} | - | "
                  f"{f(l['calibration'], 1)} | {f(l['p50'], 2)} |")
    print("\n".join(md))
    dump_json({"rows": rows, "leaderboard": LEADERBOARD, "tier_weights": TIER_WEIGHTS}, RESULTS / "jevbench_summary.json")
    (RESULTS / "jevbench_summary.md").write_text("\n".join(md) + "\n")


if __name__ == "__main__":
    main()
