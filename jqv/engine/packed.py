"""D2: packed branches. [prefix | q1 | q2 | ... | qQ] is one sequence; a block attention
mask lets each question see the prefix and itself only. Position ids restart at len(prefix)
for every branch, so each branch is numerically the same computation as prefix+q_i alone.

    prefix  ████████
    q1      ████████ ██
    q2      ████████    ██
    q3      ████████       ██

If the packed sequence exceeds `max_tokens`, the prefix is prefilled into a KV cache once
and the questions are packed in chunks against it (same mask logic, query rows = questions).
`isolate=False` gives the plain-causal packed variant (branches leak into later ones); it
exists only as the negative control for the isolation experiment.
"""

from __future__ import annotations

import torch

from jqv.engine.base import DecisionEngine


def branch_layout(prefix_len: int, branch_lens: list[int], include_prefix_rows: bool):
    """Return (branch_q, pos_q, branch_k, pos_k). branch -1 = prefix; pos = absolute index in
    the packed [prefix | branches] sequence."""
    bk, pk = [-1] * prefix_len, list(range(prefix_len))
    cur = prefix_len
    for b, n in enumerate(branch_lens):
        bk += [b] * n
        pk += list(range(cur, cur + n))
        cur += n
    if include_prefix_rows:
        bq, pq = bk, pk
    else:
        bq, pq = bk[prefix_len:], pk[prefix_len:]
    return bq, pq, bk, pk


def build_block_mask(
    prefix_len: int,
    branch_lens: list[int],
    include_prefix_rows: bool,
    dtype: torch.dtype,
    device: torch.device,
    isolate: bool = True,
) -> torch.Tensor:
    bq, pq, bk, pk = branch_layout(prefix_len, branch_lens, include_prefix_rows)
    bq = torch.tensor(bq, device=device)[:, None]
    pq = torch.tensor(pq, device=device)[:, None]
    bk = torch.tensor(bk, device=device)[None, :]
    pk = torch.tensor(pk, device=device)[None, :]
    causal = pk <= pq
    allowed = (bk == -1) & causal  # prefix: causal for prefix rows, fully visible for branch rows
    allowed |= (bk == bq) & (bq >= 0) & causal  # own branch, causal
    if not isolate:
        allowed |= (bk >= 0) & (bk < bq)  # negative control: see earlier branches
    mask = torch.zeros(allowed.shape, dtype=dtype, device=device)
    mask.masked_fill_(~allowed, torch.finfo(dtype).min)
    return mask[None, None]


def branch_position_ids(prefix_len: int, branch_lens: list[int], include_prefix: bool, isolate: bool = True) -> list[int]:
    if not isolate:  # plain causal packing: ordinary sequential positions
        start = 0 if include_prefix else prefix_len
        return list(range(start, prefix_len + sum(branch_lens)))
    pos = list(range(prefix_len)) if include_prefix else []
    for n in branch_lens:
        pos += list(range(prefix_len, prefix_len + n))
    return pos


class PackedEngine(DecisionEngine):
    name = "packed"

    def __init__(self, rt, temperature=None, max_tokens: int = 16384, use_prefix_cache: bool = False,
                 isolate: bool = True, readout: str = "full"):
        super().__init__(rt, temperature, readout)
        self.max_tokens = max_tokens
        self.use_prefix_cache = use_prefix_cache
        self.isolate = isolate

    def _readout_index(self, offset: int, branch_lens: list[int]) -> torch.Tensor:
        idx, cur = [], offset
        for n in branch_lens:
            cur += n
            idx.append(cur - 1)
        return torch.tensor(idx, device=self.rt.device)

    def _single_forward(self, prefix, suffixes):
        lens = [len(s) for s in suffixes]
        ids = self._ids(prefix + [t for s in suffixes for t in s])[None]
        if len(suffixes) == 1:
            # one branch == plain causal sequence; skip the explicit (L x L) mask so sdpa can use its causal kernel
            h = self.rt.backbone(input_ids=ids, use_cache=False).last_hidden_state
            return h[0, -1:]
        mask = build_block_mask(len(prefix), lens, True, self.rt.dtype, self.rt.device, self.isolate)
        pos = torch.tensor(branch_position_ids(len(prefix), lens, True, self.isolate), device=self.rt.device)[None]
        h = self.rt.backbone(input_ids=ids, attention_mask=mask, position_ids=pos, use_cache=False).last_hidden_state
        return h[0, self._readout_index(len(prefix), lens)]

    def _cached_forward(self, prefix, suffixes):
        s_len = len(prefix)
        cache = self.rt.backbone(input_ids=self._ids(prefix)[None], use_cache=True).past_key_values
        outs, i = [], 0
        while i < len(suffixes):
            j, total = i, 0
            while j < len(suffixes) and (j == i or total + len(suffixes[j]) <= self.max_tokens):
                total += len(suffixes[j])
                j += 1
            chunk = suffixes[i:j]
            lens = [len(s) for s in chunk]
            ids = self._ids([t for s in chunk for t in s])[None]
            mask = build_block_mask(s_len, lens, False, self.rt.dtype, self.rt.device, self.isolate)
            pos = torch.tensor(branch_position_ids(s_len, lens, False, self.isolate), device=self.rt.device)[None]
            h = self.rt.backbone(
                input_ids=ids, attention_mask=mask, position_ids=pos, past_key_values=cache, use_cache=True
            ).last_hidden_state
            outs.append(h[0, self._readout_index(0, lens)])
            cache.crop(-sum(lens))  # restore the prefix-only cache for the next chunk
            i = j
        return torch.cat(outs, dim=0)

    def readout_hidden(self, prefix, suffixes):
        total = len(prefix) + sum(len(s) for s in suffixes)
        if not self.use_prefix_cache and total <= self.max_tokens:
            return self._single_forward(prefix, suffixes)
        return self._cached_forward(prefix, suffixes)
