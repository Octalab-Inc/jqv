"""Vocabulary readout: final hidden states -> choice logits -> probabilities."""

from __future__ import annotations

import math

import torch

from jqv.types import Decision


def confidence_from_probs(p: torch.Tensor) -> float:
    """1 - H(p) / log K. 1.0 = one-hot, 0.0 = uniform."""
    k = p.numel()
    if k <= 1:
        return 1.0
    ent = -(p * (p.clamp_min(1e-12)).log()).sum().item()
    return max(0.0, 1.0 - ent / math.log(k))


@torch.inference_mode()
def decisions_from_hidden(
    hidden: torch.Tensor,
    lm_head: torch.nn.Module,
    choice_ids_per_q: list[list[int]],
    temperature: float | None = None,
    chunk: int = 256,
) -> list[Decision]:
    """hidden: (Q, H) readout hidden states (post final norm). Returns one Decision per row."""
    out: list[Decision] = []
    w_dtype = lm_head.weight.dtype
    for start in range(0, hidden.shape[0], chunk):
        h = hidden[start : start + chunk].to(w_dtype)
        logits = lm_head(h).float()  # (b, V)
        full_probs = logits.softmax(dim=-1)
        for i in range(logits.shape[0]):
            ids = torch.tensor(choice_ids_per_q[start + i], device=logits.device)
            z = logits[i, ids]
            p = z.softmax(dim=-1)
            cal = (z / temperature).softmax(dim=-1) if temperature else None
            out.append(
                Decision(
                    logits=z.tolist(),
                    probabilities=p.tolist(),
                    calibrated_probabilities=cal.tolist() if cal is not None else None,
                    confidence=confidence_from_probs(cal if cal is not None else p),
                    choice_mass=full_probs[i, ids].sum().item(),
                )
            )
    return out
