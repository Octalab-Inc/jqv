"""Decision back-ends for jqgrep: an in-process jqv runtime (default) or a running jqv server (`--server`)."""

from __future__ import annotations

import json
import urllib.request
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


class LocalClient:
    """Loads the model once and answers batches of questions against a shared state with a jqv engine."""

    def __init__(self, model_id: str | None = None, engine: str = "packed", device: str | None = None, dtype: str | None = None,
                 temperature: str | float | None = "auto", max_tokens: int = 16384):
        from jqv.engine import make_engine
        from jqv.model import load_runtime

        self.rt = load_runtime(model_id, device, dtype)
        self.model_id = self.rt.model_id
        self.engine_name = engine
        T = self._resolve_temperature(temperature)
        self.temperature = T
        kw = {"max_tokens": max_tokens} if engine == "packed" else {}
        self.engine = make_engine(engine, self.rt, T, **kw)

    def _resolve_temperature(self, temperature):
        if temperature is None:
            return None
        if temperature != "auto":
            return float(temperature)
        slug = self.rt.model_id.split("/")[-1].lower()
        for name in (f"mmlu_packed_{slug}_temperature.json",):
            path = REPO_ROOT / "results" / name
            if path.exists():
                try:
                    from jqv.calibration import TemperatureScaler
                    from jqv.prompt import prompt_hash
                    from jqv.model import default_style_for

                    ts = TemperatureScaler.load(path)
                    ts.check_compatible(self.rt.model_id, prompt_hash(default_style_for(self.rt.model_id)))
                    return ts.temperature
                except Exception:
                    return None
        return None

    def n_tokens(self, text: str) -> int:
        return len(self.rt.prompt.prefix_ids(text))

    def decide(self, state: str, questions: list[tuple[str, list[str]]]) -> list[list[float]]:
        from jqv.types import Question

        decs = self.engine.decide(state, [Question(question=q, choices=c) for q, c in questions])
        return [list(d.calibrated_probabilities or d.probabilities) for d in decs]


class HTTPClient:
    """Uses a running jqv server's POST /decision (any model / engine the server was started with)."""

    def __init__(self, base_url: str, timeout: float = 600.0):
        self.base = base_url.rstrip("/")
        self.timeout = timeout
        try:
            with urllib.request.urlopen(f"{self.base}/health", timeout=10) as r:
                h = json.loads(r.read().decode())
            self.model_id = h.get("model", "?")
            self.engine_name = h.get("engine", "?")
            self.temperature = (h.get("calibration") or {}).get("temperature")
        except Exception as e:  # pragma: no cover - network
            raise SystemExit(f"jqv server not reachable at {self.base}: {e}")

    def n_tokens(self, text: str) -> int:
        return max(1, len(text) // 3)  # rough: the server does not expose its tokenizer

    def decide(self, state: str, questions: list[tuple[str, list[str]]]) -> list[list[float]]:
        body = json.dumps({"state": state, "questions": [{"question": q, "choices": c} for q, c in questions]}).encode()
        req = urllib.request.Request(f"{self.base}/decision", data=body, headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=self.timeout) as r:
            out = json.loads(r.read().decode())
        return [list(d.get("calibrated_probabilities") or d["probabilities"]) for d in out["decisions"]]
