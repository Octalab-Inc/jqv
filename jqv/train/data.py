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
