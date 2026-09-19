from __future__ import annotations

import torch

from jqv.model import Runtime
from jqv.readout import decisions_from_hidden
from jqv.types import Decision, Question


class DecisionEngine:
    """Base class. Subclasses implement `readout_hidden` (attention structure);
    the vocabulary readout is shared."""

    name = "base"

    def __init__(self, rt: Runtime, temperature: float | None = None):
        self.rt = rt
        self.temperature = temperature

    def encode(self, state: str, questions: list[Question]) -> tuple[list[int], list[list[int]], list[list[int]]]:
        prefix = self.rt.prompt.prefix_ids(state)
        suffixes = [self.rt.prompt.suffix_ids(q.question, q.choices) for q in questions]
        choice_ids = [self.rt.prompt.choice_token_ids(len(q.choices)) for q in questions]
        return prefix, suffixes, choice_ids

    def readout_hidden(self, prefix: list[int], suffixes: list[list[int]]) -> torch.Tensor:
        """Return (Q, H) hidden states at the readout position of each question."""
        raise NotImplementedError

    @torch.inference_mode()
    def decide(self, state: str, questions: list[Question]) -> list[Decision]:
        prefix, suffixes, choice_ids = self.encode(state, questions)
        hidden = self.readout_hidden(prefix, suffixes)
        return decisions_from_hidden(hidden, self.rt.lm_head, choice_ids, self.temperature)

    # ----- helpers -----
    def _ids(self, seq: list[int]) -> torch.Tensor:
        return torch.tensor(seq, dtype=torch.long, device=self.rt.device)

    @staticmethod
    def pad_right(seqs: list[list[int]], pad_id: int) -> tuple[torch.Tensor, torch.Tensor, list[int]]:
        lens = [len(s) for s in seqs]
        t = max(lens)
        ids = torch.full((len(seqs), t), pad_id, dtype=torch.long)
        mask = torch.zeros((len(seqs), t), dtype=torch.long)
        for i, s in enumerate(seqs):
            ids[i, : len(s)] = torch.tensor(s)
            mask[i, : len(s)] = 1
        return ids, mask, lens
