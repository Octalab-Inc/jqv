"""FastAPI Decision API.

    POST /decision  {"state": "...", "questions": [{"question": "...", "choices": ["..", ".."]}]}
    -> {"engine": "packed", "decisions": [{"probabilities": [...], "calibrated_probabilities": [...], ...}]}

Configuration via env: JQV_MODEL, JQV_ENGINE (naive|kvcache|packed|generate), JQV_TEMPERATURE_FILE,
JQV_DTYPE, JQV_DEVICE. Or run `python -m jqv.server --model ... --engine ...`.
"""

from __future__ import annotations

import os
from contextlib import asynccontextmanager

from fastapi import FastAPI

from jqv.calibration import TemperatureScaler
from jqv.engine import make_engine
from jqv.model import load_runtime
from jqv.types import DecisionRequest, DecisionResponse

_state: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    rt = load_runtime(os.environ.get("JQV_MODEL"), os.environ.get("JQV_DEVICE"), os.environ.get("JQV_DTYPE"))
    temp = None
    if os.environ.get("JQV_TEMPERATURE_FILE"):
        temp = TemperatureScaler.load(os.environ["JQV_TEMPERATURE_FILE"]).temperature
    _state["rt"] = rt
    _state["engine"] = make_engine(os.environ.get("JQV_ENGINE", "packed"), rt, temp)
    yield
    _state.clear()


app = FastAPI(title="jqv Decision API", lifespan=lifespan)


@app.get("/health")
def health():
    rt = _state.get("rt")
    return {"ok": rt is not None, "model": getattr(rt, "model_id", None),
            "engine": getattr(_state.get("engine"), "name", None), "device": str(getattr(rt, "device", None))}


@app.post("/decision", response_model=DecisionResponse)
def decision(req: DecisionRequest) -> DecisionResponse:
    eng = _state["engine"]
    return DecisionResponse(
        engine=eng.name,
        model=_state["rt"].model_id,
        temperature=eng.temperature or 1.0,
        decisions=eng.decide(req.state, req.questions),
    )


def main() -> None:
    import argparse

    import uvicorn

    ap = argparse.ArgumentParser()
    ap.add_argument("--model")
    ap.add_argument("--engine", default="packed")
    ap.add_argument("--temperature-file")
    ap.add_argument("--dtype")
    ap.add_argument("--device")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    a = ap.parse_args()
    for k, v in {"JQV_MODEL": a.model, "JQV_ENGINE": a.engine, "JQV_TEMPERATURE_FILE": a.temperature_file,
                 "JQV_DTYPE": a.dtype, "JQV_DEVICE": a.device}.items():
        if v:
            os.environ[k] = v
    uvicorn.run("jqv.server:app", host=a.host, port=a.port)


if __name__ == "__main__":
    main()
