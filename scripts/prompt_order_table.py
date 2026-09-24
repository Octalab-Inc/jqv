"""Prompt-order ablation table: JevBench public tiers (accuracy, top-label ECE, Brier, ordinal MAE), hard tier by family,
paired exact McNemar vs the state_first layout on the same items, and MMLU accuracy / temperature / ECE per layout.

    uv run python scripts/prompt_order_table.py --model qwen3-32b [--layouts state_first repeat_question query_first query_first_only]

Reads results/jevbench/<model>_packed_T_ablate-<layout>/<tier>/{summary.json,results.jsonl} and
results/mmlu_packed_<model>[_ablate-a | _layout-<layout>]{.json,_calibration.json}; writes results/prompt_order_ablation_<model>.{md,json}.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from math import comb
from pathlib import Path

from _common import RESULTS, dump_json

TIERS = ("easy", "standard", "hard")
LAYOUTS = ("state_first", "repeat_question", "query_first", "query_first_only")
SHORT = {"state_first": "A state→Q", "repeat_question": "B state→Q→Q", "query_first": "C Q→state→Q", "query_first_only": "D Q→state"}


def mcnemar(a: dict, b: dict) -> tuple[int, int, float]:
    won = sum(1 for t in a if t in b and not a[t]["correct"] and b[t]["correct"])
    lost = sum(1 for t in a if t in b and a[t]["correct"] and not b[t]["correct"])
    n, k = won + lost, min(won, lost)
    p = min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2 ** n) if n else 1.0
    return won, lost, p


def load_tier(label: str, tier: str):
    d = RESULTS / "jevbench" / label / tier
    if not (d / "summary.json").exists():
        return None, None
    s = json.loads((d / "summary.json").read_text())
    rows = {}
    for line in (d / "results.jsonl").read_text().splitlines():
        if line.strip():
            r = json.loads(line)
            rows[r["task_id"]] = r
    return s, rows


def f3(x):
    return "-" if x is None else f"{x:.3f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="qwen3-32b")
    ap.add_argument("--layouts", nargs="+", default=list(LAYOUTS))
    ap.add_argument("--mmlu-tag-a", default=None, help="MMLU tag for state_first (default packed_<model>_ablate-a)")
    a = ap.parse_args()
    out = {"model": a.model, "layouts": {}}
    md = [f"### Prompt order ablation ({a.model}, zero-shot packed, served T fitted per layout on MMLU val 400)", ""]
    tiers_rows, fam_rows, mc_rows = {}, {}, {}
    ref = {}
    for L in a.layouts:
        label = f"{a.model}_packed_T_ablate-{L}"
        info = {"label": label, "tiers": {}}
        for tier in TIERS:
            s, rows = load_tier(label, tier)
            if s is None:
                continue
            e = s.get("ece")
            e = e.get("ece") if isinstance(e, dict) else e
            info["tiers"][tier] = {"accuracy": s["accuracy"], "ece": e, "brier": s.get("brier_mean"), "ordinal_mae": s.get("ordinal_mae"),
                                   "n": len(rows), "correct": sum(1 for r in rows.values() if r["correct"])}
            if L == "state_first":
                ref[tier] = rows
            elif tier in ref:
                won, lost, p = mcnemar(ref[tier], rows)
                info["tiers"][tier]["vs_state_first"] = {"won": won, "lost": lost, "p": p}
            if tier == "hard":
                fam = defaultdict(lambda: [0, 0])
                for r in rows.values():
                    fam[r["family"]][1] += 1
                    fam[r["family"]][0] += int(r["correct"])
                info["families"] = {k: v for k, v in sorted(fam.items())}
        # MMLU
        tag = (a.mmlu_tag_a or f"packed_{a.model}_ablate-a") if L == "state_first" else f"packed_{a.model}_layout-{L}"
        j = RESULTS / f"mmlu_{tag}.json"
        if j.exists():
            m = json.loads(j.read_text())
            c = json.loads((RESULTS / f"mmlu_{tag}_calibration.json").read_text()) if (RESULTS / f"mmlu_{tag}_calibration.json").exists() else {}
            info["mmlu"] = {"accuracy": m["accuracy"], "prompt_hash": m.get("prompt_hash"), "temperature": c.get("temperature"),
                            "nll_raw": (c.get("before") or {}).get("nll"), "nll_T": (c.get("after") or {}).get("nll"),
                            "ece_raw": (c.get("before") or {}).get("ece"), "ece_T": (c.get("after") or {}).get("ece"),
                            "questions_per_sec": m.get("questions_per_sec")}
        out["layouts"][L] = info
    # tables
    md.append("| layout | shared prefix | easy | standard | hard | hard vs A (won/lost, p) | hard ECE | hard Brier | hard ordinal MAE | standard ECE |")
    md.append("|---|:---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for L in a.layouts:
        t = out["layouts"][L]["tiers"]
        if not t:
            continue
        h, st, ea = t.get("hard", {}), t.get("standard", {}), t.get("easy", {})
        vs = h.get("vs_state_first")
        md.append(f"| {SHORT[L]} | {'yes' if L in ('state_first', 'repeat_question') else 'no'} | {f3(ea.get('accuracy'))} | {f3(st.get('accuracy'))} ({st.get('correct')}/{st.get('n')}) | "
                  f"**{f3(h.get('accuracy'))}** ({h.get('correct')}/{h.get('n')}) | {'-' if not vs else f'{vs[chr(119)+chr(111)+chr(110)]}/{vs[chr(108)+chr(111)+chr(115)+chr(116)]}, p={vs[chr(112)]:.3f}'} | "
                  f"{f3(h.get('ece'))} | {f3(h.get('brier'))} | {f3(h.get('ordinal_mae'))} | {f3(st.get('ece'))} |")
    fams = sorted({f for L in a.layouts for f in out["layouts"][L].get("families", {})})
    if fams:
        md += ["", "| layout | " + " | ".join(fams) + " | all |", "|---|" + "---:|" * (len(fams) + 1)]
        for L in a.layouts:
            fm = out["layouts"][L].get("families")
            if fm:
                tot = sum(v[0] for v in fm.values()); n = sum(v[1] for v in fm.values())
                md.append(f"| {SHORT[L]} | " + " | ".join(f"{fm[f][0]}/{fm[f][1]}" if f in fm else "-" for f in fams) + f" | {tot}/{n} |")
    md += ["", "| layout | MMLU acc (800) | T (MMLU val) | NLL raw / +T | ECE raw / +T | q/s (packed, 1 q/state) |", "|---|---:|---:|---:|---:|---:|"]
    for L in a.layouts:
        m = out["layouts"][L].get("mmlu")
        if m:
            md.append(f"| {SHORT[L]} | {m['accuracy']:.3f} | {f3(m['temperature'])} | {f3(m['nll_raw'])} / {f3(m['nll_T'])} | {f3(m['ece_raw'])} / {f3(m['ece_T'])} | {f3(m['questions_per_sec'])} |")
    text = "\n".join(md) + "\n"
    print(text)
    (RESULTS / f"prompt_order_ablation_{a.model}.md").write_text(text)
    dump_json(out, RESULTS / f"prompt_order_ablation_{a.model}.json")


if __name__ == "__main__":
    main()
