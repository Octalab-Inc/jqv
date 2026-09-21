"""Per-family accuracy on the JevBench hard tier from results/jevbench/<label>/hard/results.jsonl.

    uv run python scripts/jevbench_families.py                      # all labels
    uv run python scripts/jevbench_families.py --labels qwen3-32b_packed_T qwen3-14b_packed_T --json out.json
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

from _common import RESULTS

FAMILY_ORDER = ["long_policy", "multi_hop", "temporal_numeric", "probability", "tradeoff", "ambiguous", "judge_hard",
                "adversarial", "trap", "routing_hard"]


def family_table(run_dir: Path) -> dict[str, dict]:
    src = run_dir / "hard" / "results.jsonl"
    if not src.exists():
        return {}
    by = defaultdict(lambda: {"n": 0, "correct": 0})
    for line in src.read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        fam = r.get("family", "?")
        by[fam]["n"] += 1
        by[fam]["correct"] += 1 if r.get("correct") else 0
    for fam, d in by.items():
        d["accuracy"] = d["correct"] / d["n"] if d["n"] else None
    return dict(by)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", nargs="*", default=None, help="run labels under results/jevbench (default: all with a hard tier)")
    ap.add_argument("--json", default=None, help="write the table as JSON")
    a = ap.parse_args()
    root = RESULTS / "jevbench"
    labels = a.labels or sorted(p.name for p in root.iterdir() if (p / "hard" / "results.jsonl").exists())
    tables = {l: family_table(root / l) for l in labels}
    tables = {l: t for l, t in tables.items() if t}
    fams = [f for f in FAMILY_ORDER if any(f in t for t in tables.values())]
    fams += sorted({f for t in tables.values() for f in t} - set(fams))
    head = "| run | " + " | ".join(fams) + " | all |"
    print(head)
    print("|---|" + "---:|" * (len(fams) + 1))
    for l, t in tables.items():
        n_all = sum(d["n"] for d in t.values())
        c_all = sum(d["correct"] for d in t.values())
        cells = [f"{t[f]['correct']}/{t[f]['n']}" if f in t else "-" for f in fams]
        print(f"| {l} | " + " | ".join(cells) + f" | {c_all}/{n_all} = {c_all / n_all:.3f} |")
    if a.json:
        Path(a.json).write_text(json.dumps(tables, indent=2))
        print(f"wrote {a.json}")


if __name__ == "__main__":
    main()
