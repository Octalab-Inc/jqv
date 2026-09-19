"""Dataset loading. Every item is {"state": str, "question": str, "choices": [str], "answer": int}."""

from __future__ import annotations

import json
import random
from pathlib import Path

Item = dict


def load_jsonl(path: str | Path) -> list[Item]:
    items = []
    for line in Path(path).read_text().splitlines():
        if line.strip():
            d = json.loads(line)
            d.setdefault("state", "")
            items.append(d)
    return items


def load_mmlu(split: str = "test") -> list[Item]:
    from datasets import load_dataset

    ds = load_dataset("cais/mmlu", "all", split=split)
    return [
        {"state": "", "question": r["question"], "choices": list(r["choices"]), "answer": int(r["answer"]),
         "subject": r["subject"]}
        for r in ds
    ]


def load_jmmlu() -> list[Item]:
    """JMMLU (nlp-waseda/JMMLU) ships as a zip of per-subject CSVs (question,A,B,C,D,answer)."""
    import csv
    import io
    import zipfile

    from huggingface_hub import hf_hub_download

    path = hf_hub_download("nlp-waseda/JMMLU", "JMMLU.zip", repo_type="dataset")
    items = []
    with zipfile.ZipFile(path) as z:
        for name in sorted(z.namelist()):
            if not (name.startswith("JMMLU/test/") and name.endswith(".csv")):
                continue
            subject = Path(name).stem
            text = z.read(name).decode("utf-8-sig")
            for r in csv.DictReader(io.StringIO(text)):
                if not r.get("question") or r.get("answer") not in "ABCD":
                    continue
                items.append({"state": "", "question": r["question"], "choices": [r["A"], r["B"], r["C"], r["D"]],
                              "answer": "ABCD".index(r["answer"]), "subject": subject})
    return items


def load_named(name: str) -> list[Item]:
    if name == "mmlu":
        return load_mmlu()
    if name == "jmmlu":
        return load_jmmlu()
    if name == "bridge":
        return load_jsonl(Path(__file__).resolve().parent.parent / "data" / "bridge_synth.jsonl")
    if name.endswith(".jsonl"):
        return load_jsonl(name)
    raise ValueError(f"unknown dataset {name!r}")


def sample_split(items: list[Item], n: int | None, n_val: int, seed: int = 0) -> tuple[list[Item], list[Item]]:
    """Shuffle, take up to n items, and split the first n_val off as the calibration (val) set."""
    rng = random.Random(seed)
    items = list(items)
    rng.shuffle(items)
    if n:
        items = items[:n]
    return items[:n_val], items[n_val:]
