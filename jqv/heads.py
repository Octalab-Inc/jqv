"""Learned decision heads (phase C). Both read final hidden states of the same prompt B uses.

SlotHead    (C1): z = W h_d + b, one row per option slot (Hume's "slot head"). Optionally initialised from the
                  LM head rows of the letters " A", " B", ... so that step 0 equals readout B'.
PointerHead (C2): z_i = (U h_d) . (V h_i) / sqrt(r) + w . h_i, a low-rank bilinear score between the decision
                  representation h_d (last prompt token) and each option's representation h_i (last token of
                  the option text). K is variable, no slot vocabulary, permutation-equivariant by construction.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import torch
from torch import nn


class SlotHead(nn.Module):
    kind = "slot"

    def __init__(self, hidden: int, max_choices: int = 26):
        super().__init__()
        self.proj = nn.Linear(hidden, max_choices)
        nn.init.normal_(self.proj.weight, std=0.02)
        nn.init.zeros_(self.proj.bias)

    def init_from_lm_head(self, lm_head: nn.Module, letter_ids: list[int]) -> None:
        with torch.no_grad():
            self.proj.weight[: len(letter_ids)] = lm_head.weight[letter_ids].float()
            if getattr(lm_head, "bias", None) is not None:
                self.proj.bias[: len(letter_ids)] = lm_head.bias[letter_ids].float()

    def forward(self, h_d: torch.Tensor, h_opts: torch.Tensor, opt_mask: torch.Tensor) -> torch.Tensor:
        z = self.proj(h_d.float())[:, : opt_mask.shape[1]]
        return z.masked_fill(~opt_mask, float("-inf"))


class PointerHead(nn.Module):
    kind = "pointer"

    def __init__(self, hidden: int, rank: int = 256):
        super().__init__()
        self.u = nn.Linear(hidden, rank, bias=False)
        self.v = nn.Linear(hidden, rank, bias=False)
        self.w = nn.Linear(hidden, 1, bias=False)
        self.scale = 1.0 / math.sqrt(rank)
        for m in (self.u, self.v, self.w):
            nn.init.normal_(m.weight, std=0.02)

    def forward(self, h_d: torch.Tensor, h_opts: torch.Tensor, opt_mask: torch.Tensor) -> torch.Tensor:
        qd = self.u(h_d.float())  # (N, r)
        ko = self.v(h_opts.float())  # (N, K, r)
        z = torch.einsum("nr,nkr->nk", qd, ko) * self.scale + self.w(h_opts.float()).squeeze(-1)
        return z.masked_fill(~opt_mask, float("-inf"))


def make_head(kind: str, hidden: int, **kw) -> nn.Module:
    if kind == "slot":
        return SlotHead(hidden, **kw)
    if kind == "pointer":
        return PointerHead(hidden, **kw)
    raise ValueError(kind)


def save_head(head: nn.Module, path: Path, extra: dict | None = None) -> None:
    path.mkdir(parents=True, exist_ok=True)
    cfg = {"kind": head.kind, "hidden": (head.proj.in_features if head.kind == "slot" else head.u.in_features)}
    if head.kind == "slot":
        cfg["max_choices"] = head.proj.out_features
    else:
        cfg["rank"] = head.u.out_features
    cfg.update(extra or {})
    (path / "head.json").write_text(json.dumps(cfg, indent=2))
    torch.save(head.state_dict(), path / "head.pt")


def load_head(path: Path, device) -> tuple[nn.Module, dict]:
    cfg = json.loads((path / "head.json").read_text())
    kw = {"max_choices": cfg["max_choices"]} if cfg["kind"] == "slot" else {"rank": cfg["rank"]}
    head = make_head(cfg["kind"], cfg["hidden"], **kw)
    head.load_state_dict(torch.load(path / "head.pt", map_location="cpu"))
    return head.to(device).eval(), cfg
