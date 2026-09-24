from __future__ import annotations

import torch

from jqv.model import Runtime
from jqv.readout import _decision, decisions_from_hidden
from jqv.types import Decision, Question


class DecisionEngine:
    """Base class. Subclasses implement `readout_hidden` (attention structure);
    the vocabulary readout is shared."""

    name = "base"

    def __init__(self, rt: Runtime, temperature: float | None = None, readout: str = "full", perm_avg: bool = False):
        self.rt = rt
        self.temperature = temperature
        self.readout = readout  # "full" (B: full LM head) or "rows" (B': letters' rows only)
        # perm_avg: evaluate every cyclic rotation of the option order (K per question) and average the
        # probabilities, which removes position / letter priors at K x the per-question cost (the state is shared).
        self.perm_avg = perm_avg

    def encode(self, state: str, questions: list[Question]) -> tuple[list[int], list[list[int]], list[list[int]]]:
        prefix = self.rt.prompt.prefix_ids(state)
        suffixes = [self.rt.prompt.suffix_ids(q.question, q.choices, q.labels) for q in questions]
        choice_ids = [self.rt.prompt.choice_token_ids(len(q.choices), q.labels) for q in questions]
        return prefix, suffixes, choice_ids

    def readout_hidden(self, prefix: list[int], suffixes: list[list[int]]) -> torch.Tensor:
        """Return (Q, H) hidden states at the readout position of each question."""
        raise NotImplementedError

    @torch.inference_mode()
    def decide(self, state: str, questions: list[Question]) -> list[Decision]:
        if self.perm_avg:
            return self._decide_perm_avg(state, questions)
        return self._decide_plain(state, questions)

    def _decide_plain(self, state: str, questions: list[Question]) -> list[Decision]:
        p = self.rt.prompt
        if p.per_question_prefix:
            # query-first layouts: the state comes after the question, so every question has its own prefix and
            # nothing is shared across questions (one forward per question, whatever the engine).
            hidden = torch.cat([self.readout_hidden(p.prefix_ids(state, q.question, q.choices, q.labels),
                                                    [p.suffix_ids(q.question, q.choices, q.labels)]) for q in questions], 0)
            choice_ids = [p.choice_token_ids(len(q.choices), q.labels) for q in questions]
        else:
            prefix, suffixes, choice_ids = self.encode(state, questions)
            hidden = self.readout_hidden(prefix, suffixes)
        return decisions_from_hidden(hidden, self.rt.lm_head, choice_ids, self.temperature, readout=self.readout)

    def _decide_perm_avg(self, state: str, questions: list[Question]) -> list[Decision]:
        """All K cyclic rotations of each question's options in one call, mapped back and averaged.
        The averaged probabilities become the decision; `logits` = log(mean p) so temperature scaling still applies."""
        variants, index = [], []
        for qi, q in enumerate(questions):
            k = len(q.choices)
            for s in range(k):
                sem = [(i + s) % k for i in range(k)]  # position i shows original choice sem[i]
                variants.append(Question(question=q.question, choices=[q.choices[j] for j in sem]))
                index.append((qi, sem))
        plain = self._decide_plain(state, variants)
        acc: dict[int, list] = {}
        for (qi, sem), d in zip(index, plain):
            p = torch.tensor(d.probabilities)
            sem_p = torch.zeros(len(sem))
            for pos, j in enumerate(sem):
                sem_p[j] = p[pos]
            acc.setdefault(qi, []).append((sem_p, d.choice_mass))
        out: list[Decision] = [None] * len(questions)  # type: ignore[list-item]
        for qi, lst in acc.items():
            mean_p = torch.stack([x[0] for x in lst]).mean(0)
            masses = [x[1] for x in lst if x[1] is not None]
            dec = _decision(mean_p.clamp_min(1e-12).log(), self.temperature, sum(masses) / len(masses) if masses else None)
            dec.perm_avg_k = len(lst)
            out[qi] = dec
        return out

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
