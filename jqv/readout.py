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
            ref = cal if cal is not None else p
            out.append(
                Decision(
                    logits=z.tolist(),
                    probabilities=p.tolist(),
                    calibrated_probabilities=cal.tolist() if cal is not None else None,
                    confidence=confidence_from_probs(ref),
                    entropy_concentration=entropy_concentration(ref),
                    choice_mass=full_probs[i, ids].sum().item(),
                )
            )
    return out
