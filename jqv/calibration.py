"""Temperature scaling: p = softmax(z / T), T fitted by minimizing NLL on held-out data."""

from __future__ import annotations

import json
import math
import os
import warnings
from datetime import datetime, timezone
from pathlib import Path

import torch

META_FIELDS = ("model", "engine", "prompt_hash", "dataset", "n_val", "choice_counts", "fitted_at", "dtype")


class CalibrationMismatch(RuntimeError):
    pass


class TemperatureScaler:
    """A fitted temperature plus provenance: which model / prompt / dataset it was fitted on.

    A temperature is only meaningful for the distribution it was fitted on. `meta` records that so the
    server can refuse (or warn about) a temperature fitted for a different model or prompt layout.
    """

    def __init__(self, temperature: float = 1.0, meta: dict | None = None):
        self.temperature = float(temperature)
        self.meta = dict(meta or {})

    def fit(self, logits: torch.Tensor, labels: torch.Tensor, t_min: float = 0.05, t_max: float = 100.0,
            grid: int = 400, refine: int = 40) -> "TemperatureScaler":
        """Minimize NLL over T in [t_min, t_max] (log-spaced grid, then golden-section refinement).

        logits: (N, Kmax) with -inf padding for unused choices; labels: (N,) int.
        A bounded 1-D search is used instead of LBFGS: on tiny or near-separable sets (e.g. 30 items,
        all confidently correct) the NLL keeps improving as T -> 0 and an unbounded optimizer diverges.
        """
        z = logits.float().cpu()
        y = labels.long().cpu()

        def nll(log_t: float) -> float:
            return torch.nn.functional.cross_entropy(z / math.exp(log_t), y).item()

        lo, hi = math.log(t_min), math.log(t_max)
        pts = torch.linspace(lo, hi, grid).tolist()
        best = min(pts, key=nll)
        i = pts.index(best)
        a, b = pts[max(i - 1, 0)], pts[min(i + 1, grid - 1)]
        phi = (math.sqrt(5) - 1) / 2
        c, d = b - phi * (b - a), a + phi * (b - a)
        fc, fd = nll(c), nll(d)
        for _ in range(refine):
            if fc < fd:
                b, d, fd = d, c, fc
                c = b - phi * (b - a)
                fc = nll(c)
            else:
                a, c, fc = c, d, fd
                d = a + phi * (b - a)
                fd = nll(d)
        self.temperature = float(math.exp((a + b) / 2))
        return self

    def apply(self, logits: torch.Tensor) -> torch.Tensor:
        return (logits.float() / self.temperature).softmax(dim=-1)

    def save(self, path: str | Path, **meta) -> None:
        """Write {"temperature": T, ...meta}. Extra keyword args are merged into the stored metadata."""
        self.meta.update({k: v for k, v in meta.items() if v is not None})
        self.meta.setdefault("fitted_at", datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"))
        Path(path).write_text(json.dumps({"temperature": self.temperature, **self.meta}, indent=2, ensure_ascii=False))

    @classmethod
    def load(cls, path: str | Path) -> "TemperatureScaler":
        """Reads both the current format and the legacy {"temperature": T} format (meta empty)."""
        d = json.loads(Path(path).read_text())
        meta = {k: v for k, v in d.items() if k != "temperature"}
        return cls(d["temperature"], meta)

    def check_compatible(self, model_id: str, prompt_hash: str, allow_mismatch: bool | None = None) -> list[str]:
        """Compare stored provenance with the runtime. Returns the list of mismatches; raises
        CalibrationMismatch unless allow_mismatch (default: env JQV_ALLOW_CALIBRATION_MISMATCH=1)."""
        if allow_mismatch is None:
            allow_mismatch = os.environ.get("JQV_ALLOW_CALIBRATION_MISMATCH", "") == "1"
        problems = []
        if self.meta.get("model") and self.meta["model"] != model_id:
            problems.append(f"model: fitted on {self.meta['model']!r}, runtime is {model_id!r}")
        if self.meta.get("prompt_hash") and self.meta["prompt_hash"] != prompt_hash:
            problems.append(f"prompt_hash: fitted on {self.meta['prompt_hash']}, runtime is {prompt_hash}")
        if not self.meta.get("model") or not self.meta.get("prompt_hash"):
            problems.append("no provenance recorded (legacy temperature file); cannot verify model / prompt")
        if problems:
            msg = "temperature file is not verified for this runtime: " + "; ".join(problems)
            if not allow_mismatch:
                raise CalibrationMismatch(msg + " (set JQV_ALLOW_CALIBRATION_MISMATCH=1 to use it anyway)")
            warnings.warn(msg)
        return problems

    def info(self) -> dict:
        return {"temperature": self.temperature, **{k: self.meta.get(k) for k in META_FIELDS}}

    def __repr__(self) -> str:
        src = self.meta.get("dataset")
        return f"TemperatureScaler(T={self.temperature:.4f}" + (f", fitted on {src})" if src else ")")


def logit_from_prob(p: float) -> float:
    return math.log(p) - math.log1p(-p)
