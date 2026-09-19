"""D1: shared KV cache. The prefix (state) is prefilled once; questions are run as a
right-padded batch against a batch-replicated copy of the prefix cache.

Memory note: HF DynamicCache materializes the prefix K/V once per batch row, so the
batch size is auto-limited by `cache_budget_bytes`. The packed engine (D2) avoids this
entirely because all branches share one copy of the prefix K/V inside a single sequence.
"""

from __future__ import annotations

import copy

import torch

from jqv.engine.base import DecisionEngine


class KVCacheEngine(DecisionEngine):
    name = "kvcache"

    def __init__(self, rt, temperature=None, batch_size: int = 64, cache_budget_bytes: int = 8 << 30):
        super().__init__(rt, temperature)
        self.batch_size = batch_size
        self.cache_budget_bytes = cache_budget_bytes

    def _prefill(self, prefix: list[int]):
        out = self.rt.backbone(input_ids=self._ids(prefix)[None], use_cache=True)
        return out.past_key_values

    def _cache_bytes(self, cache) -> int:
        total = 0
        for layer in cache.layers:
            if getattr(layer, "is_initialized", False):
                total += layer.keys.numel() * layer.keys.element_size() * 2
        return max(total, 1)

    def readout_hidden(self, prefix, suffixes):
        pad_id = self.rt.tokenizer.pad_token_id or 0
        cache = self._prefill(prefix)
        s_len = len(prefix)
        bs = max(1, min(self.batch_size, self.cache_budget_bytes // self._cache_bytes(cache)))
        outs = []
        for start in range(0, len(suffixes), bs):
            chunk = suffixes[start : start + bs]
            ids, mask, lens = self.pad_right(chunk, pad_id)
            b, t = ids.shape
            c = copy.deepcopy(cache)
            c.batch_repeat_interleave(b)
            attn = torch.cat([torch.ones((b, s_len), dtype=torch.long), mask], dim=1).to(self.rt.device)
            pos = (torch.arange(t) + s_len)[None].expand(b, t).to(self.rt.device)
            h = self.rt.backbone(
                input_ids=ids.to(self.rt.device),
                attention_mask=attn,
                position_ids=pos,
                past_key_values=c,
                use_cache=True,
            ).last_hidden_state
            idx = torch.tensor([l - 1 for l in lens], device=h.device)
            outs.append(h[torch.arange(b, device=h.device), idx])
            del c
        return torch.cat(outs, dim=0)
