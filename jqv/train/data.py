"""Training examples for decision heads: MMLU auxiliary_train / validation with option-order shuffling."""

from __future__ import annotations

import random
from dataclasses import dataclass

import torch


@dataclass
class Example:
    ids: list[int]  # prefix + suffix token ids
    opt_ends: list[int]  # absolute index of the last token of each option's text
    answer: int


def load_split(split: str) -> list[dict]:
    from datasets import load_dataset

    ds = load_dataset("cais/mmlu", "all", split=split)
    return [{"question": r["question"], "choices": list(r["choices"]), "answer": int(r["answer"])} for r in ds]


def load_source(name: str) -> list[dict]:
    """A training/validation source: 'mmlu' (auxiliary_train), 'mmlu_val' (validation), 'synth:<family>' (train split)
    or 'synth:<family>:<split>'. Items are {"state", "question", "choices", "answer"}."""
    if name == "mmlu":
        return load_split("auxiliary_train")
    if name == "mmlu_val":
        return load_split("validation")
    if name.startswith("synth:"):
        from jqv.data import load_synth

        parts = name.split(":")
        family, split = parts[1], (parts[2] if len(parts) > 2 else "train")
        return load_synth(family, split)
    raise ValueError(f"unknown source {name!r}")


def parse_mix(spec: str) -> list[tuple[str, float]]:
    """'synth:long_policy=0.25,synth:probability=0.2,mmlu=0.3' -> [(name, weight), ...] (weights need not sum to 1)."""
    out = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        name, _, w = part.partition("=")
        out.append((name.strip(), float(w) if w else 1.0))
    if not out:
        raise ValueError("empty mix spec")
    return out


class MixSampler:
    """Draws each batch from ONE source chosen by weight, so a batch has homogeneous lengths (no 300-token MMLU items
    padded to a 3k-token policy document). Each source is cycled through in a seeded shuffled order."""

    def __init__(self, sources: dict[str, list], weights: dict[str, float], rng: random.Random):
        self.names = [n for n in sources if sources[n]]
        self.weights = [weights[n] for n in self.names]
        self.items = {n: list(sources[n]) for n in self.names}
        self.rng = rng
        for n in self.names:
            rng.shuffle(self.items[n])
        self.cursor = {n: 0 for n in self.names}
        self.last_source: str | None = None

    def next(self, n: int, make) -> list:
        src = self.rng.choices(self.names, weights=self.weights)[0]
        self.last_source = src
        items, out = self.items[src], []
        tries = 0
        while len(out) < n and tries < 20 * n:
            it = items[self.cursor[src] % len(items)]
            self.cursor[src] += 1
            tries += 1
            e = make(it)
            if e is not None:
                out.append(e)
        return out


def make_example(prompt, item: dict, rng: random.Random | None, max_len: int) -> Example | None:
    choices, answer = list(item["choices"]), item["answer"]
    if rng is not None:  # shuffle option order so heads cannot rely on position
        perm = list(range(len(choices)))
        rng.shuffle(perm)
        choices = [item["choices"][j] for j in perm]
        answer = perm.index(item["answer"])
    prefix = prompt.prefix_ids(item.get("state", ""))
    suffix, ends = prompt.suffix_ids_with_spans(item["question"], choices)
    if len(prefix) + len(suffix) > max_len:
        return None
    return Example(prefix + suffix, [len(prefix) + e for e in ends], answer)


def collate(examples: list[Example], pad_id: int, device):
    n = len(examples)
    t = max(len(e.ids) for e in examples)
    k = max(len(e.opt_ends) for e in examples)
    ids = torch.full((n, t), pad_id, dtype=torch.long)
    attn = torch.zeros((n, t), dtype=torch.long)
    dec = torch.zeros(n, dtype=torch.long)
    opt = torch.zeros((n, k), dtype=torch.long)
    opt_mask = torch.zeros((n, k), dtype=torch.bool)
    y = torch.zeros(n, dtype=torch.long)
    for i, e in enumerate(examples):
        ids[i, : len(e.ids)] = torch.tensor(e.ids)
        attn[i, : len(e.ids)] = 1
        dec[i] = len(e.ids) - 1
        opt[i, : len(e.opt_ends)] = torch.tensor(e.opt_ends)
        opt_mask[i, : len(e.opt_ends)] = True
        y[i] = e.answer
    return {k_: v.to(device) for k_, v in dict(ids=ids, attn=attn, dec=dec, opt=opt, opt_mask=opt_mask, y=y).items()}


def gather_features(hidden: torch.Tensor, batch: dict) -> tuple[torch.Tensor, torch.Tensor]:
    """hidden (N, T, H) -> h_d (N, H) at the decision position, h_opts (N, K, H) at option ends."""
    n = hidden.shape[0]
    ar = torch.arange(n, device=hidden.device)
    h_d = hidden[ar, batch["dec"]]
    h_opts = hidden[ar[:, None], batch["opt"]]
    return h_d, h_opts
