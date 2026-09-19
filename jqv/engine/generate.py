"""A: ordinary generation baseline. The model generates a few tokens; we parse the letter.
Probabilities are one-hot (or uniform if unparseable)."""

from __future__ import annotations

import re

import torch

from jqv.engine.base import DecisionEngine
from jqv.prompt import LETTERS
from jqv.readout import confidence_from_probs, entropy_concentration
from jqv.types import Decision, Question


class GenerateEngine(DecisionEngine):
    name = "generate"

    def __init__(self, rt, temperature=None, batch_size: int = 8, max_new_tokens: int = 4, readout: str = "full"):
        super().__init__(rt, temperature)  # readout is irrelevant: generation needs the full vocabulary
        self.batch_size = batch_size
        self.max_new_tokens = max_new_tokens

    @torch.inference_mode()
    def decide(self, state: str, questions: list[Question]) -> list[Decision]:
        prefix, suffixes, _ = self.encode(state, questions)
        tok = self.rt.tokenizer
        pad_id = tok.pad_token_id or 0
        out: list[Decision] = []
        for start in range(0, len(suffixes), self.batch_size):
            qs = questions[start : start + self.batch_size]
            seqs = [prefix + s for s in suffixes[start : start + self.batch_size]]
            # left padding for generation
            t = max(len(s) for s in seqs)
            ids = torch.full((len(seqs), t), pad_id, dtype=torch.long)
            mask = torch.zeros((len(seqs), t), dtype=torch.long)
            for i, s in enumerate(seqs):
                ids[i, t - len(s) :] = torch.tensor(s)
                mask[i, t - len(s) :] = 1
            gen = self.rt.model.generate(
                input_ids=ids.to(self.rt.device),
                attention_mask=mask.to(self.rt.device),
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                pad_token_id=pad_id,
            )
            texts = tok.batch_decode(gen[:, t:], skip_special_tokens=True)
            for q, text in zip(qs, texts):
                k = len(q.choices)
                labels = self.rt.prompt.resolve_labels(k, q.labels)
                m = re.search(r"[A-Z]", text)
                idx = labels.index(m.group(0)) if (m and m.group(0) in labels) else -1
                if 0 <= idx < k:
                    p = [0.0] * k
                    p[idx] = 1.0
                    parsed = True
                else:
                    p = [1.0 / k] * k
                    parsed = False
                out.append(
                    Decision(
                        logits=[0.0] * k,
                        probabilities=p,
                        confidence=confidence_from_probs(torch.tensor(p)),
                        entropy_concentration=entropy_concentration(torch.tensor(p)),
                        parsed=parsed,
                    )
                )
        return out
