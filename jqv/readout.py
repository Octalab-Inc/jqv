"""Vocabulary readout: final hidden states -> choice logits -> probabilities."""

from __future__ import annotations

import math

import torch

from jqv.types import Decision


def confidence_from_probs(p: torch.Tensor) -> float:
    """Jev-compatible confidence: (p_max - 1/K) / (1 - 1/K). 1.0 = one-hot, 0.0 = uniform.

    This is the post-hoc formula Hume confirmed in TypeSafe's official adapter. It summarizes how far the
    distribution is from uniform; it is not itself a probability of being correct.
    """
    k = p.numel()
    if k <= 1:
        return 1.0
    return max(0.0, (p.max().item() - 1.0 / k) / (1.0 - 1.0 / k))


def entropy_concentration(p: torch.Tensor) -> float:
    """1 - H(p) / log K. 1.0 = one-hot, 0.0 = uniform. Entropy-based alternative to `confidence`."""
    k = p.numel()
    if k <= 1:
        return 1.0
    ent = -(p * (p.clamp_min(1e-12)).log()).sum().item()
    return max(0.0, 1.0 - ent / math.log(k))


READOUT_MODES = ("full", "rows")


def _decision(z: torch.Tensor, temperature: float | None, choice_mass: float | None) -> Decision:
    p = z.softmax(dim=-1)
    cal = (z / temperature).softmax(dim=-1) if temperature else None
    ref = cal if cal is not None else p
    return Decision(
        logits=z.tolist(),
        probabilities=p.tolist(),
        calibrated_probabilities=cal.tolist() if cal is not None else None,
        confidence=confidence_from_probs(ref),
        entropy_concentration=entropy_concentration(ref),
        choice_mass=choice_mass,
    )


@torch.inference_mode()
def decisions_from_hidden(
    hidden: torch.Tensor,
    lm_head: torch.nn.Module,
    choice_ids_per_q: list[list[int]],
    temperature: float | None = None,
    chunk: int = 256,
    readout: str = "full",
) -> list[Decision]:
    """hidden: (Q, H) readout hidden states (post final norm). Returns one Decision per row.

    readout="full" (B):  z = lm_head(h)[choice_ids]  -- full-vocabulary projection, then pick the letters.
                          Also yields `choice_mass`, the full-vocab softmax mass on the letters.
    readout="rows" (B'): z = h @ W[choice_ids].T + b[choice_ids] -- only the letters' rows of the LM head.
                          Mathematically identical logits; no (Q, V) projection; `choice_mass` is None.
    """
    if readout not in READOUT_MODES:
        raise ValueError(f"readout must be one of {READOUT_MODES}, got {readout!r}")
    if readout == "rows":
        return _decisions_from_rows(hidden, lm_head, choice_ids_per_q, temperature)
    out: list[Decision] = []
    w_dtype = lm_head.weight.dtype
    for start in range(0, hidden.shape[0], chunk):
        h = hidden[start : start + chunk].to(w_dtype)
        logits = lm_head(h).float()  # (b, V)
        full_probs = logits.softmax(dim=-1)
        for i in range(logits.shape[0]):
            ids = torch.tensor(choice_ids_per_q[start + i], device=logits.device)
            out.append(_decision(logits[i, ids], temperature, full_probs[i, ids].sum().item()))
    return out


@torch.inference_mode()
def _decisions_from_rows(hidden, lm_head, choice_ids_per_q, temperature) -> list[Decision]:
    uniq = sorted({t for ids in choice_ids_per_q for t in ids})
    col = {t: j for j, t in enumerate(uniq)}
    idx = torch.tensor(uniq, device=hidden.device)
    w = lm_head.weight[idx]  # (U, H): only the letters' rows
    z_all = (hidden.to(w.dtype) @ w.T).float()  # (Q, U)
    if getattr(lm_head, "bias", None) is not None:
        z_all = z_all + lm_head.bias[idx].float()
    out: list[Decision] = []
    for i, ids in enumerate(choice_ids_per_q):
        cols = torch.tensor([col[t] for t in ids], device=z_all.device)
        out.append(_decision(z_all[i, cols], temperature, None))
    return out
