"""Generate synthetic items: python -m jqv.synth.generate --family temporal_numeric --n-train 2000 --n-dev 300 --n-test 500

Splits use separate RNG streams (seed, seed+1, seed+2) and one shared signature registry, so the same scenario
parameters never appear in two splits. Progress is printed every 200 items.
"""

from __future__ import annotations

import argparse
import importlib
import json
import random
import sys
import time
from pathlib import Path

from jqv.synth import FAMILIES
from jqv.synth.common import Names, SplitWriter

MODULES = {"temporal_numeric": "jqv.synth.temporal", "probability": "jqv.synth.probability", "long_policy": "jqv.synth.policy",
           "temporal_v2": "jqv.synth.temporal_v2"}


def generate_family(family: str, counts: dict[str, int], seed: int, out_dir: Path, max_tries_factor: int = 20,
                    apply_paraphrase: bool = True) -> dict:
    mod = importlib.import_module(MODULES[family])
    writer = SplitWriter(family, out_dir, seed)
    t0 = time.time()
    for i, (split, n) in enumerate(counts.items()):
        rng = random.Random(seed * 1000 + i)
        names = Names(rng)
        made, tries = 0, 0
        while made < n and tries < n * max_tries_factor:
            tries += 1
            try:
                item = mod.make_item(rng, names)
                item.validate()
            except Exception as e:  # a generator bug or an unsatisfiable draw: report and continue
                if tries <= 5 or tries % 500 == 0:
                    print(f"  [{family}/{split}] skipped ({type(e).__name__}: {e})", file=sys.stderr)
                continue
            if writer.add(split, item):
                made += 1
                if made % 200 == 0 or made == n:
                    print(f"  [{family}/{split}] {made}/{n} ({tries} tries, {time.time() - t0:.0f}s)", flush=True)
        if made < n:
            print(f"  [{family}/{split}] WARNING only {made}/{n} unique items after {tries} tries", file=sys.stderr)
    paths = writer.write()
    if apply_paraphrase:
        from jqv.synth.paraphrase import apply_patch

        for split, p in paths.items():
            n = apply_patch(p)
            if n:
                print(f"  [{family}/{split}] re-applied paraphrase patch to {n} items")
    summary = writer.summary()
    (out_dir / family / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    for split, p in paths.items():
        print(f"wrote {p} ({len(writer.items[split])} items)")
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--family", default="all", help="temporal_numeric | probability | long_policy | temporal_v2 | all")
    ap.add_argument("--n-train", type=int, default=2000)
    ap.add_argument("--n-dev", type=int, default=300)
    ap.add_argument("--n-test", type=int, default=500)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="data/synth")
    ap.add_argument("--no-paraphrase-patch", action="store_true", help="do not re-apply <split>.paraphrase.jsonl after generating")
    a = ap.parse_args()
    fams = list(FAMILIES) if a.family == "all" else [a.family]
    counts = {"train": a.n_train, "dev": a.n_dev, "test": a.n_test}
    for fam in fams:
        print(f"== {fam}: {counts} seed={a.seed}")
        s = generate_family(fam, counts, a.seed, Path(a.out), apply_paraphrase=not a.no_paraphrase_patch)
        for split, d in s.items():
            print(f"  {split}: n={d['n']} hops={d['hops_hist']} qtype={d['qtype']} tokens={d['state_tokens']}")


if __name__ == "__main__":
    main()
