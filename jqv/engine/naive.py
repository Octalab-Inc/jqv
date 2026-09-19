"""B: one full forward per question (prefix + suffix), batched with right padding.
Baseline for 'no shared computation'."""

from __future__ import annotations

import torch

from jqv.engine.base import DecisionEngine


class NaiveEngine(DecisionEngine):
    name = "naive"

    def __init__(self, rt, temperature=None, batch_size: int = 8, readout: str = "full"):
        super().__init__(rt, temperature, readout)
        self.batch_size = batch_size

    def readout_hidden(self, prefix, suffixes):
        pad_id = self.rt.tokenizer.pad_token_id or 0
        outs = []
        for start in range(0, len(suffixes), self.batch_size):
            chunk = [prefix + s for s in suffixes[start : start + self.batch_size]]
            ids, mask, lens = self.pad_right(chunk, pad_id)
            ids = ids.to(self.rt.device)
            # no padding -> no mask, so sdpa can use its causal kernel instead of an explicit (L x L) mask
            mask = mask.to(self.rt.device) if len(set(lens)) > 1 else None
            h = self.rt.backbone(input_ids=ids, attention_mask=mask, use_cache=False).last_hidden_state
            idx = torch.tensor([l - 1 for l in lens], device=h.device)
            outs.append(h[torch.arange(h.shape[0], device=h.device), idx])
        return torch.cat(outs, dim=0)
