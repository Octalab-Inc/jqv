import pytest

from jqv.engine import make_engine
from jqv.types import Question


def _group(items, state):
    return [Question(question=i["question"], choices=i["choices"]) for i in items if i["state"] == state]


@pytest.mark.parametrize("engine,kwargs", [
    ("kvcache", {}),
    ("kvcache", {"batch_size": 2}),
    ("packed", {}),
    ("packed", {"use_prefix_cache": True}),
    ("packed", {"use_prefix_cache": True, "chunk_tokens": 120}),  # forces chunking
    ("shared", {}),
    ("shared", {"backend": "manual"}),
    ("shared", {"use_prefix_cache": True}),
    ("shared", {"use_prefix_cache": True, "chunk_tokens": 120}),  # forces chunking
])
def test_engine_matches_naive(rt, bridge_items, engine, kwargs):
    state = bridge_items[0]["state"]
    qs = _group(bridge_items, state)
    assert len(qs) >= 3
    ref = make_engine("naive", rt).decide(state, qs)
    out = make_engine(engine, rt, **kwargs).decide(state, qs)
    fp32 = rt.dtype.itemsize == 4
    # fp32: exact up to accumulation order. bf16: logits of magnitude 32-128 have a spacing of 0.25-0.5, so two
    # forward paths can differ by a couple of ulps (observed: exactly 0.5 at Qwen3-14B); probabilities within 5e-2.
    tol_logit, tol_prob = (1e-2, 1e-3) if fp32 else (1.0, 5e-2)
    for a, b in zip(ref, out):
        assert len(a.logits) == len(b.logits)
        assert max(abs(x - y) for x, y in zip(a.logits, b.logits)) <= tol_logit
        assert max(abs(x - y) for x, y in zip(a.probabilities, b.probabilities)) < tol_prob


@pytest.mark.parametrize("engine", ["naive", "kvcache", "packed", "shared"])
def test_rows_readout_matches_full(rt, bridge_items, engine):
    """B' (letters' LM-head rows only) must give the same choice logits as B (full-vocab projection)."""
    state = bridge_items[0]["state"]
    qs = _group(bridge_items, state)
    full = make_engine(engine, rt, readout="full").decide(state, qs)
    rows = make_engine(engine, rt, readout="rows").decide(state, qs)
    tol = 1e-4 if rt.dtype.itemsize == 4 else 5e-2
    for a, b in zip(full, rows):
        assert max(abs(x - y) for x, y in zip(a.logits, b.logits)) < tol
        assert a.choice_mass is not None and b.choice_mass is None
