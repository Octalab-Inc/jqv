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


def load_synth(family: str, split: str, root: str | Path | None = None) -> list[Item]:
    """Synthetic JevBench-shaped items (data/synth/<family>/<split>.jsonl) as jqv items.

    The wire-format question (choice / noul / score) is turned into choices exactly as the TypeSafe-compatible
    server does it, so the model sees the same text in eval and in serving. Extra keys: id, family, labels,
    qtype, target_distribution, meta (dependency_hops, reasoning_depth, ...).
    """
    from jqv.systemone import build_question

    root = Path(root) if root else Path(__file__).resolve().parent.parent / "data" / "synth"
    path = root / family / f"{split}.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"{path} (run: uv run python -m jqv.synth.generate --family {family})")
    items = []
    for rec in load_jsonl(path):
        q, meta = build_question(rec["id"], rec["question"])
        items.append({
            "state": rec["state"], "question": q.question, "choices": q.choices,
            "answer": meta["labels"].index(rec["expected"]), "id": rec["id"], "family": rec["family"],
            "labels": meta["labels"], "qtype": meta["type"], "target_distribution": rec.get("target_distribution"),
            "meta": rec.get("meta", {}), "subject": rec.get("meta", {}).get("scenario"),
        })
    return items


def load_named(name: str) -> list[Item]:
    if name == "mmlu":
        return load_mmlu()
    if name == "jmmlu":
        return load_jmmlu()
    if name == "bridge":
        return load_jsonl(Path(__file__).resolve().parent.parent / "data" / "bridge_synth.jsonl")
    if name.startswith("synth:"):
        parts = name.split(":")
        if len(parts) != 3:
            raise ValueError("synthetic datasets are named synth:<family>:<split>")
        return load_synth(parts[1], parts[2])
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
