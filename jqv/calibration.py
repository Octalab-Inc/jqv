"""Temperature scaling: p = softmax(z / T), T fitted by minimizing NLL on held-out data."""

from __future__ import annotations

import json
import math
from pathlib import Path

import torch


class TemperatureScaler:
    def __init__(self, temperature: float = 1.0):
        self.temperature = float(temperature)

    def fit(self, logits: torch.Tensor, labels: torch.Tensor, max_iter: int = 200) -> "TemperatureScaler":
        """logits: (N, Kmax) with -inf padding for unused choices; labels: (N,) int."""
        logits = logits.float().cpu()
        labels = labels.long().cpu()
        log_t = torch.zeros((), requires_grad=True)
        opt = torch.optim.LBFGS([log_t], lr=0.1, max_iter=max_iter, line_search_fn="strong_wolfe")

        def closure():
            opt.zero_grad()
            loss = torch.nn.functional.cross_entropy(logits / log_t.exp(), labels)
            loss.backward()
            return loss

        opt.step(closure)
        self.temperature = float(log_t.exp().item())
        return self

    def apply(self, logits: torch.Tensor) -> torch.Tensor:
        return (logits.float() / self.temperature).softmax(dim=-1)

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps({"temperature": self.temperature}, indent=2))

    @classmethod
    def load(cls, path: str | Path) -> "TemperatureScaler":
        return cls(json.loads(Path(path).read_text())["temperature"])

    def __repr__(self) -> str:
        return f"TemperatureScaler(T={self.temperature:.4f})"


def logit_from_prob(p: float) -> float:
    return math.log(p) - math.log1p(-p)
