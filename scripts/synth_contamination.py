"""Check synthetic data against the JevBench public items: shared word 8-grams, reused IDs, reused invented names.

    uv run python scripts/synth_contamination.py --jevbench /Users/h.imura/tmp/repo/jevbench/datasets/public [--synth data/synth]

Exit code 1 if more than 1% of synthetic items share an 8-gram with a JevBench item or if any JevBench ID pattern
or capitalised name pair from JevBench appears in the synthetic states.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

WORD = re.compile(r"[a-z0-9]+")
ID_PAT = re.compile(r"\b[A-Z]{2,5}-\d{2,}(?:-\d+)?\b")
NAME_PAIR = re.compile(r"\b([A-Z][a-z]{2,})\s([A-Z][a-z]{2,})\b")
COMMON = {"The", "This", "That", "These", "Where", "Under", "Part", "Section", "Policy", "Claim", "Form", "Note", "Page",
          "Named", "Insured", "Residence", "Premises", "Delivery", "Product", "Term", "Extended", "Warranty", "Certificate",
          "Extract", "And", "For", "With", "From", "Prepared", "Bracketed", "Coverage", "Review", "Current", "Edition",
          "Exclusion", "Exception", "Definition", "Definitions", "Endorsement", "Endorsements", "Company", "Group", "Home",
          "Appliances", "Electronics", "Incoming", "Inspection", "Supplier", "Sampling", "Plan", "Version", "Lot", "Units"}


def text_of(rec: dict) -> str:
    st = rec.get("state", "")
    if not isinstance(st, str):
        st = json.dumps(st, ensure_ascii=False)
    q = rec.get("question", "")
    if not isinstance(q, str):
        q = json.dumps(q, ensure_ascii=False)
    return st + "\n" + q


def ngrams(text: str, n: int = 8) -> set[tuple[str, ...]]:
    w = WORD.findall(text.lower())
    return {tuple(w[i:i + n]) for i in range(len(w) - n + 1)}


def load(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jevbench", required=True, help="directory with easy.jsonl / original.jsonl / hard.jsonl")
    ap.add_argument("--synth", default="data/synth")
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--show", type=int, default=5)
    a = ap.parse_args()
    jb = []
    for name in ("easy", "original", "hard"):
        p = Path(a.jevbench) / f"{name}.jsonl"
        if p.exists():
            jb += load(p)
    jb_text = "\n".join(text_of(r) for r in jb)
    jb_ngrams = set()
    for r in jb:
        jb_ngrams |= ngrams(text_of(r), a.n)
    jb_ids = set(ID_PAT.findall(jb_text))
    jb_names = {m for m in NAME_PAIR.findall(jb_text) if m[0] not in COMMON and m[1] not in COMMON}
    jb_names = {" ".join(m) for m in jb_names}
    print(f"JevBench: {len(jb)} items, {len(jb_ngrams):,} distinct {a.n}-grams, {len(jb_ids)} IDs, {len(jb_names)} name pairs")

    bad = False
    for fam_dir in sorted(Path(a.synth).iterdir()):
        if not fam_dir.is_dir():
            continue
        for split in ("train", "dev", "test"):
            p = fam_dir / f"{split}.jsonl"
            if not p.exists():
                continue
            recs = load(p)
            hits, id_hits, name_hits, shared = 0, Counter(), Counter(), Counter()
            for r in recs:
                t = text_of(r)
                g = ngrams(t, a.n) & jb_ngrams
                if g:
                    hits += 1
                    for x in list(g)[:3]:
                        shared[" ".join(x)] += 1
                for i in set(ID_PAT.findall(t)) & jb_ids:
                    id_hits[i] += 1
                for nm in jb_names:
                    if nm in t:
                        name_hits[nm] += 1
            frac = hits / max(1, len(recs))
            flag = "FAIL" if (frac > 0.01 or id_hits or name_hits) else "ok"
            print(f"{fam_dir.name}/{split}: n={len(recs)} items with shared {a.n}-grams: {hits} ({frac:.2%}), ID reuse: {sum(id_hits.values())}, "
                  f"name reuse: {sum(name_hits.values())} -> {flag}")
            for s, c in shared.most_common(a.show):
                print(f"    shared: '{s}' x{c}")
            for s, c in list(id_hits.items())[:5] + list(name_hits.items())[:5]:
                print(f"    reused: {s} x{c}")
            bad |= flag == "FAIL"
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
