"""Shared pieces: derivation traces, item assembly in the JevBench wire shape, split bookkeeping, text helpers."""

from __future__ import annotations

import hashlib
import json
import random
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

GENERATOR_VERSION = "0.1"
NOUL_LABELS = ("no", "yes")


class Trace:
    """A derivation DAG. `given` facts come straight from the state; derived steps depend on earlier steps.

    dependency_hops = number of derived steps; reasoning_depth = longest chain of derived steps.
    """

    def __init__(self) -> None:
        self.steps: list[tuple[str, tuple[str, ...], str, bool]] = []  # (name, deps, text, derived)
        self._names: set[str] = set()

    def given(self, name: str, text: str) -> str:
        self._add(name, (), text, derived=False)
        return name

    def derive(self, name: str, deps: Iterable[str], text: str) -> str:
        deps = tuple(deps)
        for d in deps:
            if d not in self._names:
                raise KeyError(f"trace step {name!r} depends on unknown step {d!r}")
        self._add(name, deps, text, derived=True)
        return name

    def _add(self, name, deps, text, derived):
        if name in self._names:
            raise ValueError(f"duplicate trace step {name!r}")
        self._names.add(name)
        self.steps.append((name, deps, text, derived))

    @property
    def hops(self) -> int:
        return sum(1 for _, _, _, d in self.steps if d)

    @property
    def depth(self) -> int:
        longest: dict[str, int] = {}
        for name, deps, _, derived in self.steps:
            best = max((longest[d] for d in deps), default=0)
            longest[name] = best + (1 if derived else 0)
        return max(longest.values(), default=0)

    def rationale(self) -> str:
        out = []
        for i, (_, _, text, derived) in enumerate(s for s in self.steps if s[3]):
            out.append(f"({i + 1}) {text}")
        return " ".join(out)


@dataclass
class SynthItem:
    family: str
    scenario: str
    state: str
    qtype: str  # choice | noul | score
    instructions: str
    criteria: dict | list  # choice: {label: desc}; noul: {"true": .., "false": ..}; score: [desc0, desc1, ..]
    expected: str
    trace: Trace
    signature: str
    distractor: str = ""
    surface_answer: str = ""
    target_distribution: dict | None = None
    extra: dict = field(default_factory=dict)

    @property
    def labels(self) -> list[str]:
        if self.qtype == "choice":
            return list(self.criteria.keys())
        if self.qtype == "noul":
            return list(NOUL_LABELS)
        return [str(i) for i in range(len(self.criteria))]

    def validate(self) -> None:
        if self.qtype not in ("choice", "noul", "score"):
            raise ValueError(self.qtype)
        labels = self.labels
        if self.expected not in labels:
            raise ValueError(f"expected {self.expected!r} not in labels {labels}")
        if len(labels) < 2 or len(set(labels)) != len(labels):
            raise ValueError(f"bad labels {labels}")
        if self.qtype == "choice" and not all(re.fullmatch(r"[a-z0-9_]+", l) for l in labels):
            raise ValueError(f"choice labels must be snake_case: {labels}")
        if self.target_distribution is not None:
            if set(self.target_distribution) != set(labels):
                raise ValueError("target_distribution keys must equal labels")
            s = sum(self.target_distribution.values())
            if abs(s - 1.0) > 1e-6:
                raise ValueError(f"target_distribution sums to {s}")
        if self.surface_answer and self.surface_answer not in labels:
            raise ValueError(f"surface_answer {self.surface_answer!r} not in labels")
        if not self.state.strip() or not self.instructions.strip():
            raise ValueError("empty state or instructions")

    def to_record(self, item_id: str, split: str, seed: int, state_tokens: int) -> dict:
        self.validate()
        question = {"type": self.qtype, "instructions": self.instructions, "criteria": self.criteria}
        return {
            "id": item_id,
            "family": self.family,
            "split": split,
            "state": self.state,
            "question": question,
            "labels": self.labels,
            "expected": self.expected,
            "target_distribution": self.target_distribution,
            "rationale": self.trace.rationale(),
            "meta": {
                "scenario": self.scenario,
                "qtype": self.qtype,
                "K": len(self.labels),
                "dependency_hops": self.trace.hops,
                "reasoning_depth": self.trace.depth,
                "distractor": self.distractor,
                "surface_answer": self.surface_answer,
                "state_tokens": state_tokens,
                **self.extra,
            },
            "generator": {"name": f"jqv.synth.{self.family}", "version": GENERATOR_VERSION},
            "seed": seed,
        }


# ----------------------------------------------------------------------------- text helpers

_TOKENIZER = None


def count_tokens(text: str) -> int:
    """Qwen3 token count when the tokenizer is cached locally, otherwise a chars/4 estimate."""
    global _TOKENIZER
    if _TOKENIZER is None:
        try:
            from transformers import AutoTokenizer

            _TOKENIZER = AutoTokenizer.from_pretrained("Qwen/Qwen3-1.7B")
        except Exception:  # offline or transformers missing
            _TOKENIZER = False
    if _TOKENIZER:
        return len(_TOKENIZER(text, add_special_tokens=False)["input_ids"])
    return max(1, len(text) // 4)


def money(x, currency: str = "$") -> str:
    from decimal import Decimal

    d = Decimal(str(x)).quantize(Decimal("0.01"))
    if d == d.to_integral_value():
        return f"{currency}{int(d):,}"
    return f"{currency}{d:,.2f}"


def pct(x) -> str:
    """Percent with the shortest exact representation (0.125 -> '12.5%')."""
    v = float(x) * 100
    s = f"{v:.4f}".rstrip("0").rstrip(".")
    return s + "%"


def ordinal(n: int) -> str:
    return "%d%s" % (n, "tsnrhtdd"[(n // 10 % 10 != 1) * (n % 10 < 4) * n % 10 :: 4])


def sentence_case(s: str) -> str:
    return s[:1].upper() + s[1:]


class Names:
    """Invented names that do not appear in JevBench (checked by scripts/synth_contamination.py)."""

    FIRST = ["Aiko", "Bram", "Cerys", "Dario", "Elin", "Farid", "Greta", "Hiro", "Imani", "Jonas", "Kalinda", "Leif",
             "Maren", "Nikhil", "Orla", "Pavel", "Quinn", "Rosa", "Soren", "Talia", "Umar", "Vera", "Wendel", "Ximena",
             "Yusuf", "Zofia", "Anouk", "Bashir", "Corin", "Delphine", "Emeka", "Fenna", "Gustav", "Halima", "Ines",
             "Jiro", "Katja", "Lorcan", "Milena", "Nadir", "Oyelaran", "Petra", "Rafael", "Sanne", "Tomasz", "Ulla"]
    LAST = ["Abara", "Brandt", "Castellan", "Dvorak", "Ellery", "Fontaine", "Grieg", "Haldane", "Ishida", "Jaramillo",
            "Kessler", "Lindqvist", "Marchetti", "Nowicki", "Oduya", "Pellerin", "Quiroga", "Rasmussen", "Sato",
            "Tremblay", "Ueda", "Varga", "Whitlock", "Yilmaz", "Zanetti", "Bakker", "Coetzee", "Delacroix", "Eriksen",
            "Ferreira", "Gallo", "Hoffmann", "Iversen", "Jansen", "Kowalczyk", "Laurent", "Moreau", "Nakamura",
            "Olsen", "Pires", "Reinholt", "Sandoval", "Tanaka", "Villanueva", "Weiss", "Zubair"]
    ORG_A = ["Northgate", "Silverbrook", "Ardent", "Meridian", "Copperleaf", "Halcyon", "Ironwood", "Larkspur",
             "Oakhaven", "Pinecrest", "Redfern", "Stonebridge", "Tidewater", "Umbra", "Vantage", "Westmark",
             "Ashcombe", "Bluewater", "Cinder", "Driftwood", "Evergreen", "Foxglove", "Granite", "Hollowell",
             "Juniper", "Kestrel", "Lumen", "Marlow", "Nimbus", "Orchard", "Quarry", "Ridgeway", "Saltmarsh", "Thistle"]
    ORG_B = ["Assurance", "Mutual", "Underwriters", "Insurance Group", "Indemnity", "Cooperative", "Holdings",
             "Systems", "Logistics", "Industries", "Components", "Manufacturing", "Technologies", "Services",
             "Partners", "Engineering", "Foods", "Energy", "Textiles", "Medical"]
    CITY = [("Oslo", "Europe/Oslo"), ("Lisbon", "Europe/Lisbon"), ("Warsaw", "Europe/Warsaw"), ("Athens", "Europe/Athens"),
            ("Dublin", "Europe/Dublin"), ("Helsinki", "Europe/Helsinki"), ("Madrid", "Europe/Madrid"),
            ("Vienna", "Europe/Vienna"), ("Toronto", "America/Toronto"), ("Denver", "America/Denver"),
            ("Chicago", "America/Chicago"), ("Seattle", "America/Los_Angeles"), ("Halifax", "America/Halifax"),
            ("Mexico City", "America/Mexico_City"), ("Sao Paulo", "America/Sao_Paulo"), ("Santiago", "America/Santiago"),
            ("Osaka", "Asia/Tokyo"), ("Seoul", "Asia/Seoul"), ("Singapore", "Asia/Singapore"), ("Mumbai", "Asia/Kolkata"),
            ("Dubai", "Asia/Dubai"), ("Perth", "Australia/Perth"), ("Sydney", "Australia/Sydney"),
            ("Auckland", "Pacific/Auckland"), ("Cape Town", "Africa/Johannesburg"), ("Nairobi", "Africa/Nairobi")]
    STREET = ["Alder", "Birch", "Cedar", "Dogwood", "Elm", "Fir", "Hawthorn", "Ivy", "Juniper", "Laurel", "Maple",
              "Poplar", "Rowan", "Spruce", "Willow", "Yew"]
    STREET_TYPE = ["Lane", "Road", "Avenue", "Court", "Drive", "Crescent", "Terrace", "Way"]

    def __init__(self, rng: random.Random):
        self.rng = rng

    def person(self) -> str:
        return f"{self.rng.choice(self.FIRST)} {self.rng.choice(self.LAST)}"

    def org(self, kind: str | None = None) -> str:
        b = kind or self.rng.choice(self.ORG_B)
        return f"{self.rng.choice(self.ORG_A)} {b}"

    def city(self) -> tuple[str, str]:
        return self.rng.choice(self.CITY)

    def address(self) -> str:
        return f"{self.rng.randint(2, 480)} {self.rng.choice(self.STREET)} {self.rng.choice(self.STREET_TYPE)}"

    def ident(self, prefix: str, digits: int = 6) -> str:
        return f"{prefix}-{self.rng.randint(10 ** (digits - 1), 10 ** digits - 1)}"


# ----------------------------------------------------------------------------- split bookkeeping


class SplitWriter:
    """Collects items per split, enforces signature uniqueness across splits, writes JSONL."""

    def __init__(self, family: str, out_dir: Path, seed: int):
        self.family = family
        self.out_dir = Path(out_dir) / family
        self.seed = seed
        self.seen: set[str] = set()
        self.items: dict[str, list[dict]] = {"train": [], "dev": [], "test": []}

    def add(self, split: str, item: SynthItem) -> bool:
        sig = hashlib.sha1(f"{item.family}|{item.scenario}|{item.signature}".encode()).hexdigest()[:16]
        if sig in self.seen:
            return False
        self.seen.add(sig)
        n = len(self.items[split])
        rec = item.to_record(f"synth-{self.family}-{split}-{n:05d}", split, self.seed, count_tokens(item.state))
        rec["signature"] = sig
        self.items[split].append(rec)
        return True

    def write(self) -> dict[str, Path]:
        self.out_dir.mkdir(parents=True, exist_ok=True)
        paths = {}
        for split, recs in self.items.items():
            p = self.out_dir / f"{split}.jsonl"
            with p.open("w") as f:
                for r in recs:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
            paths[split] = p
        return paths

    def summary(self) -> dict:
        out = {}
        for split, recs in self.items.items():
            if not recs:
                continue
            hops = [r["meta"]["dependency_hops"] for r in recs]
            toks = [r["meta"]["state_tokens"] for r in recs]
            out[split] = {
                "n": len(recs),
                "hops_hist": {str(h): hops.count(h) for h in sorted(set(hops))},
                "qtype": {t: sum(1 for r in recs if r["meta"]["qtype"] == t) for t in ("choice", "noul", "score")},
                "scenarios": {s: sum(1 for r in recs if r["meta"]["scenario"] == s)
                              for s in sorted({r["meta"]["scenario"] for r in recs})},
                "state_tokens": {"min": min(toks), "median": sorted(toks)[len(toks) // 2], "max": max(toks)},
                "expected_hist": {e: sum(1 for r in recs if r["expected"] == e) for e in sorted({r["expected"] for r in recs})},
            }
        return out
