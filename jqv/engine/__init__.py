from __future__ import annotations

from jqv.engine.base import DecisionEngine
from jqv.engine.generate import GenerateEngine
from jqv.engine.kvcache import KVCacheEngine
from jqv.engine.naive import NaiveEngine
from jqv.engine.packed import PackedEngine

ENGINES: dict[str, type[DecisionEngine]] = {
    "generate": GenerateEngine,  # A
    "naive": NaiveEngine,  # B
    "kvcache": KVCacheEngine,  # D1
    "packed": PackedEngine,  # D2
}


def make_engine(name: str, rt, temperature: float | None = None, **kwargs) -> DecisionEngine:
    if name == "packed_causal":  # negative control for isolation experiments
        return PackedEngine(rt, temperature, isolate=False, **kwargs)
    try:
        cls = ENGINES[name]
    except KeyError:
        raise ValueError(f"unknown engine {name!r}; choose from {list(ENGINES) + ['packed_causal']}") from None
    return cls(rt, temperature, **kwargs)


__all__ = ["ENGINES", "make_engine", "DecisionEngine", "GenerateEngine", "NaiveEngine", "KVCacheEngine", "PackedEngine"]
