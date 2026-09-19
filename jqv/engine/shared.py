"""D3: shared-prefix attention without an (L x L) mask.

Same packed input as D2 ([prefix | q1 | q2 | ...], branch-local position ids), but attention is computed
by a custom function registered in Transformers' AttentionInterface, which uses the block structure
directly instead of masking a dense matrix:

    prefix rows   : ordinary causal attention over the prefix  (fused SDPA, no explicit mask)
    branch rows   : (a) all branch queries attend the shared prefix keys in ONE dense (sum q_i) x S block
                        -- no mask, shared by every branch (the Hydragen decomposition).
                        backend="fused": two unmasked SDPA calls; the partition function needed for the
                        merge is read off a zero "probe" key (see _prefix_block_fused). backend="manual":
                        explicit chunked matmul + softmax statistics (memory-bound, slower on MPS).
                    (b) each branch attends its own keys causally, batched over branches with padding
                        -- (Q, qmax, qmax), tiny
                    (a) and (b) are merged with the flash-attention log-sum-exp identity.

Attention work is O(S^2/2 + (sum q_i) * S + sum q_i^2) instead of O(L^2) with L = S + sum q_i, and the
largest temporary is a (rows_chunk x S) score block, never (L x L). On CUDA the same block structure can be
expressed as a FlexAttention BlockMask (`backend="flex"`, untested here: FlexAttention needs Triton).
"""

from __future__ import annotations

import contextlib
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from transformers import AttentionInterface

from jqv.engine.base import DecisionEngine
from jqv.engine.packed import branch_position_ids

IMPL_NAME = "jqv_shared"


@dataclass
class Layout:
    prefix_len: int  # shared prefix tokens in the key/value sequence
    branch_lens: list[int]  # branch lengths in this forward
    prefix_in_query: bool  # True: prefix rows are part of the query (single forward); False: prefix comes from KV cache
    # index tensors for the within-branch block, built once per forward (not once per layer)
    idx: torch.Tensor | None = None  # (Q, T) flat row index of token t of branch b (0 for padding)
    valid: torch.Tensor | None = None  # (Q, T) bool
    rows_b: torch.Tensor | None = None  # (Lb,) branch of each flat row
    rows_t: torch.Tensor | None = None  # (Lb,) position within its branch of each flat row
    causal: torch.Tensor | None = None  # (T, T) bool lower-triangular

    def build(self, device) -> "Layout":
        lens = self.branch_lens
        Q, T = len(lens), max(lens)
        starts = [0]
        for n in lens[:-1]:
            starts.append(starts[-1] + n)
        idx = torch.zeros(Q, T, dtype=torch.long)
        valid = torch.zeros(Q, T, dtype=torch.bool)
        rows_b, rows_t = [], []
        for b, n in enumerate(lens):
            idx[b, :n] = torch.arange(starts[b], starts[b] + n)
            valid[b, :n] = True
            rows_b.extend([b] * n)
            rows_t.extend(range(n))
        self.idx, self.valid = idx.to(device), valid.to(device)
        self.rows_b = torch.tensor(rows_b, dtype=torch.long, device=device)
        self.rows_t = torch.tensor(rows_t, dtype=torch.long, device=device)
        self.causal = torch.tril(torch.ones(T, T, dtype=torch.bool, device=device))
        return self


_LAYOUT: Layout | None = None  # set around each forward by SharedPrefixEngine (single process, single stream)
_BACKEND = "fused"  # "fused" (any device, default) | "manual" (any device) | "flex" (CUDA, untested)
ROWS_PER_CHUNK = 512  # manual backend: branch query rows per (rows x S) score block
FUSED_ROWS_PER_CHUNK = 8192  # fused backend: branch query rows per SDPA call
BRANCH_ELEMS_PER_CHUNK = 64 << 20  # cap on Q * T * T per kv-head group in the within-branch block


def _repeat_kv(x: torch.Tensor, g: int) -> torch.Tensor:  # (1, Hk, L, D) -> (1, Hk*g, L, D)
    if g == 1:
        return x
    b, hk, l, d = x.shape
    return x[:, :, None].expand(b, hk, g, l, d).reshape(b, hk * g, l, d)


def _partial_prefix(qb, kp, vp, scale, g):
    """Branch queries -> shared prefix keys (no mask). fp32 partial softmax stats in flat row order:
    m (H, Lb, 1), l (H, Lb, 1), o (H, Lb, D)."""
    _, H, Lb, D = qb.shape
    Hk, S = kp.shape[1], kp.shape[2]
    q = qb[0].view(Hk, g, Lb, D)
    k = kp[0].float()  # (Hk, S, D)
    v = vp[0].float()
    m = torch.empty(Hk, g, Lb, 1, dtype=torch.float32, device=qb.device)
    l = torch.empty_like(m)
    o = torch.empty(Hk, g, Lb, D, dtype=torch.float32, device=qb.device)
    for r0 in range(0, Lb, ROWS_PER_CHUNK):
        r1 = min(Lb, r0 + ROWS_PER_CHUNK)
        qc = q[:, :, r0:r1].reshape(Hk, g * (r1 - r0), D).float()
        s = torch.matmul(qc, k.transpose(1, 2)) * scale  # (Hk, g*r, S): the shared block, computed once
        mc = s.amax(-1, keepdim=True)
        e = (s - mc).exp()
        m[:, :, r0:r1] = mc.view(Hk, g, r1 - r0, 1)
        l[:, :, r0:r1] = e.sum(-1, keepdim=True).view(Hk, g, r1 - r0, 1)
        o[:, :, r0:r1] = torch.matmul(e, v).view(Hk, g, r1 - r0, D)
    return m.view(H, Lb, 1), l.view(H, Lb, 1), o.view(H, Lb, D)


KEY_PAD = 8  # pad the prefix key length to a multiple of this (MPS SDPA is several x slower on unaligned lengths)


def _prefix_block_fused(qb, kp, vp, scale, g):
    """Branch queries -> shared prefix keys with two fused SDPA calls and no mask.

    The prefix keys are padded with P >= 1 all-zero keys (score 0 for every query) up to a multiple of KEY_PAD.
    Call 1: real values (zero for the pad keys) -> o' = Z_A * o_A / (Z_A + P).
    Call 2: a value matrix that is zero except channel 0 of the first pad key -> c = 1 / (Z_A + P).
    Hence Z_A = 1/c - P and o_A = o' / (1 - P c). Both calls use the unmodified 128-dim q/k/v shapes, which
    is what MPS SDPA is fast for (a 129- or 136-dim head, a tiny value dim, or an unaligned key length each
    cost 3-10x). Call 2 costs the same as call 1, so the block is ~2x one unmasked SDPA, still well below the
    masked dense kernel. Scores are bounded (|s| < ~50 for Qwen3, which has q/k RMSNorm), so c never underflows.
    Returns fp32 logZ (H, Lb, 1) and the normalized prefix-only output o (H, Lb, D)."""
    _, H, Lb, D = qb.shape
    S = kp.shape[2]
    dev, dt = qb.device, qb.dtype
    P = KEY_PAD - (S % KEY_PAD) if S % KEY_PAD else KEY_PAD  # at least one pad key: the probe
    k = torch.zeros(1, H, S + P, D, dtype=dt, device=dev)
    v = torch.zeros(1, H, S + P, D, dtype=dt, device=dev)
    k[:, :, :S] = _repeat_kv(kp, g)
    v[:, :, :S] = _repeat_kv(vp, g)
    v_probe = torch.zeros(1, H, S + P, D, dtype=dt, device=dev)
    v_probe[:, :, S, 0] = 1.0
    logz = torch.empty(H, Lb, 1, dtype=torch.float32, device=dev)
    o = torch.empty(H, Lb, D, dtype=torch.float32, device=dev)
    for r0 in range(0, Lb, FUSED_ROWS_PER_CHUNK):
        r1 = min(Lb, r0 + FUSED_ROWS_PER_CHUNK)
        q = qb[:, :, r0:r1]
        o_pad = F.scaled_dot_product_attention(q, k, v, scale=scale)[0].float()  # Z_A o_A / (Z_A + P)
        c = F.scaled_dot_product_attention(q, k, v_probe, scale=scale)[0, :, :, :1].float()  # 1 / (Z_A + P)
        c = c.clamp(min=1e-38, max=1.0 / P)
        share = (1.0 - P * c).clamp_min(1e-6)  # Z_A / (Z_A + P)
        logz[:, r0:r1] = torch.log(share) - torch.log(c)
        o[:, r0:r1] = o_pad / share
    return logz, o


def _partial_branch(qb, kb, vb, lay: Layout, scale, g):
    """Branch queries -> own-branch keys, causal. Batched over branches with padding to T = max(lens).
    Returns fp32 partial stats in flat row order (same layout as _partial_prefix)."""
    _, H, Lb, D = qb.shape
    Hk = kb.shape[1]
    device = qb.device
    lens = lay.branch_lens
    Q, T = len(lens), max(lens)
    idx, valid, rows_b, rows_t, causal = lay.idx, lay.valid, lay.rows_b, lay.rows_t, lay.causal

    m = torch.empty(Hk, g, Lb, 1, dtype=torch.float32, device=device)
    l = torch.empty_like(m)
    o = torch.empty(Hk, g, Lb, D, dtype=torch.float32, device=device)
    qf, kf, vf = qb[0].float(), kb[0].float(), vb[0].float()
    per_branch = max(1, BRANCH_ELEMS_PER_CHUNK // (T * T))
    for b0 in range(0, Q, per_branch):
        b1 = min(Q, b0 + per_branch)
        ix, va = idx[b0:b1], valid[b0:b1]
        qp = qf[:, ix].view(Hk, g, b1 - b0, T, D)  # (Hk, g, Qc, T, D)
        kp_ = kf[:, ix]  # (Hk, Qc, T, D)
        vp_ = vf[:, ix]
        s = torch.einsum("hgqtd,hqsd->hgqts", qp, kp_) * scale  # (Hk, g, Qc, T, T)
        allowed = causal[None, None, None] & va[None, None, :, None, :]
        s = s.masked_fill(~allowed, float("-inf"))
        mc = s.amax(-1, keepdim=True)
        mc = torch.where(va[None, None, :, :, None], mc, torch.zeros_like(mc))  # padded query rows: avoid inf-inf
        e = (s - mc).exp()
        lc = e.sum(-1, keepdim=True)
        oc = torch.einsum("hgqts,hqsd->hgqtd", e, vp_)
        if b0 == 0 and b1 == Q:
            rb, rt_, sel = rows_b, rows_t, slice(None)
        else:
            sel = (rows_b >= b0) & (rows_b < b1)
            rb, rt_ = rows_b[sel] - b0, rows_t[sel]
        m[:, :, sel] = mc[:, :, rb, rt_]
        l[:, :, sel] = lc[:, :, rb, rt_]
        o[:, :, sel] = oc[:, :, rb, rt_]
    return m.view(H, Lb, 1), l.view(H, Lb, 1), o.view(H, Lb, D)


def _flex_forward(query, key, value, layout: Layout, scale: float):  # CUDA only, untested on this machine
    from torch.nn.attention.flex_attention import create_block_mask, flex_attention

    from jqv.engine.packed import branch_layout

    bq, pq, bk, pk = branch_layout(layout.prefix_len, layout.branch_lens, layout.prefix_in_query)
    dev = query.device
    bq, pq, bk, pk = (torch.tensor(x, device=dev) for x in (bq, pq, bk, pk))

    def mask_mod(b, h, q_idx, kv_idx):
        causal = pk[kv_idx] <= pq[q_idx]
        return ((bk[kv_idx] == -1) & causal) | ((bk[kv_idx] == bq[q_idx]) & causal)

    block = create_block_mask(mask_mod, None, None, query.shape[2], key.shape[2], device=str(dev))
    out = flex_attention(query, key, value, block_mask=block, scale=scale, enable_gqa=key.shape[1] != query.shape[1])
    return out.transpose(1, 2).contiguous(), None


def shared_prefix_attention_forward(module, query, key, value, attention_mask, scaling=None, dropout=0.0, **kwargs):
    lay = _LAYOUT
    if lay is None:
        raise RuntimeError("jqv_shared attention called without a Layout; use SharedPrefixEngine")
    B, H, Lq, D = query.shape
    Hk, Lk = key.shape[1], key.shape[2]
    if B != 1:
        raise ValueError("SharedPrefixEngine works on a single packed sequence (batch 1)")
    g = H // Hk
    scale = scaling if scaling is not None else D**-0.5
    S, lens = lay.prefix_len, lay.branch_lens
    Lb = sum(lens)
    q_off = S if lay.prefix_in_query else 0
    if Lq != q_off + Lb or Lk != S + Lb:
        raise ValueError(f"layout mismatch: Lq={Lq}, Lk={Lk}, S={S}, sum(branch)={Lb}, prefix_in_query={lay.prefix_in_query}")
    if _BACKEND == "flex" and query.is_cuda:
        return _flex_forward(query, key, value, lay, scale)

    out = torch.empty_like(query)  # (1, H, Lq, D)
    kp, vp = key[:, :, :S], value[:, :, :S]
    if lay.prefix_in_query:  # 1) prefix rows: plain causal attention, fused kernel, no explicit mask
        out[:, :, :S] = F.scaled_dot_product_attention(query[:, :, :S], _repeat_kv(kp, g), _repeat_kv(vp, g),
                                                       is_causal=True, scale=scale)
    if Lb:
        qb = query[:, :, q_off:]
        m3, l3, o3 = _partial_branch(qb, key[:, :, S:], value[:, :, S:], lay, scale, g)  # 3) own branch, causal
        if _BACKEND == "manual":
            m2, l2, o2 = _partial_prefix(qb, kp, vp, scale, g)  # 2) branch rows -> shared prefix (one block)
            m = torch.maximum(m2, m3)  # 4) log-sum-exp merge of unnormalized partials
            a2, a3 = (m2 - m).exp(), (m3 - m).exp()
            merged = (o2 * a2 + o3 * a3) / (l2 * a2 + l3 * a3)
        else:
            logz_a, o_a = _prefix_block_fused(qb, kp, vp, scale, g)  # 2) one fused SDPA call, no mask
            logz_b = m3 + torch.log(l3)
            w = torch.sigmoid(logz_a - logz_b)  # 4) Z_A / (Z_A + Z_B)
            merged = w * o_a + (1.0 - w) * (o3 / l3)
        out[:, :, q_off:] = merged.to(query.dtype)
    return out.transpose(1, 2).contiguous(), None


def ensure_registered() -> None:
    if IMPL_NAME not in AttentionInterface._global_mapping:
        AttentionInterface.register(IMPL_NAME, shared_prefix_attention_forward)


@contextlib.contextmanager
def attn_impl(model, name: str):
    """Temporarily switch the model's attention implementation (restored afterwards)."""
    prev = model.config._attn_implementation
    _set_impl(model, name)
    try:
        yield
    finally:
        _set_impl(model, prev)


def _set_impl(model, name: str) -> None:
    try:
        model.set_attn_implementation(name)
    except Exception:
        pass
    if model.config._attn_implementation != name:
        model.config._attn_implementation_internal = name  # config object is shared by every attention module


class SharedPrefixEngine(DecisionEngine):
    name = "shared"

    def __init__(self, rt, temperature=None, max_tokens: int = 16384, use_prefix_cache: bool = False,
                 readout: str = "full", backend: str = "fused", chunk_tokens: int | None = None):
        super().__init__(rt, temperature, readout)
        self.max_tokens = max_tokens  # single-forward threshold (prefix + all branches)
        self.chunk_tokens = chunk_tokens or max_tokens  # branch tokens per forward in the prefix-cache path
        self.use_prefix_cache = use_prefix_cache
        self.backend = backend
        ensure_registered()

    def _run(self, ids, pos, layout: Layout, cache=None):
        global _LAYOUT, _BACKEND
        _LAYOUT, _BACKEND = layout.build(self.rt.device), self.backend
        try:
            with attn_impl(self.rt.model, IMPL_NAME):
                return self.rt.backbone(input_ids=ids, position_ids=pos, past_key_values=cache,
                                        use_cache=cache is not None).last_hidden_state
        finally:
            _LAYOUT = None

    @staticmethod
    def _readout_index(offset: int, lens: list[int], device) -> torch.Tensor:
        idx, cur = [], offset
        for n in lens:
            cur += n
            idx.append(cur - 1)
        return torch.tensor(idx, device=device)

    def readout_hidden(self, prefix, suffixes):
        s_len = len(prefix)
        lens = [len(s) for s in suffixes]
        dev = self.rt.device
        if not self.use_prefix_cache and s_len + sum(lens) <= self.max_tokens:
            ids = self._ids(prefix + [t for s in suffixes for t in s])[None]
            pos = torch.tensor(branch_position_ids(s_len, lens, True), device=dev)[None]
            h = self._run(ids, pos, Layout(s_len, lens, True))
            return h[0, self._readout_index(s_len, lens, dev)]
        # prefix -> KV cache with the standard implementation, then branches in chunks against it
        cache = self.rt.backbone(input_ids=self._ids(prefix)[None], use_cache=True).past_key_values
        outs, i = [], 0
        while i < len(suffixes):
            j, total = i, 0
            while j < len(suffixes) and (j == i or total + len(suffixes[j]) <= self.chunk_tokens):
                total += len(suffixes[j])
                j += 1
            chunk = suffixes[i:j]
            clens = [len(s) for s in chunk]
            ids = self._ids([t for s in chunk for t in s])[None]
            pos = torch.tensor(branch_position_ids(s_len, clens, False), device=dev)[None]
            h = self._run(ids, pos, Layout(s_len, clens, False), cache=cache)
            outs.append(h[0, self._readout_index(0, clens, dev)])
            cache.crop(-sum(clens))
            i = j
        return torch.cat(outs, dim=0)
