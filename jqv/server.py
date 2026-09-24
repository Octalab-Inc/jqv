"""FastAPI Decision API.

    POST /decision  {"state": "...", "questions": [{"question": "...", "choices": ["..", ".."]}]}
    -> {"engine": "packed", "decisions": [{"probabilities": [...], "calibrated_probabilities": [...], ...}]}

Configuration via env: JQV_MODEL, JQV_ENGINE (naive|kvcache|packed|shared|generate|slot|pointer), JQV_TEMPERATURE_FILE,
JQV_DTYPE, JQV_DEVICE, JQV_PERM_AVG=1 (option-rotation averaging), JQV_HEAD_DIR (for slot|pointer),
JQV_LAYOUT (prompt order: state_first | repeat_question | query_first | query_first_only).
Or run `python -m jqv.server --model ... --engine ...`.

Also serves `POST /v1/systemone`, the TypeSafe-compatible wire format used by JevBench (see jqv/systemone.py).

A temperature file records the model / prompt / dataset it was fitted on. Startup fails if it was fitted
for a different model or prompt layout, unless JQV_ALLOW_CALIBRATION_MISMATCH=1 (then it warns).
"""

from __future__ import annotations

import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request

from jqv.calibration import TemperatureScaler
from jqv.engine import make_engine
from jqv.model import DEFAULT_MODEL, default_style_for, load_runtime
from jqv.prompt import prompt_hash
from jqv.systemone import SystemOneError, decide_systemone
from jqv.types import CalibrationInfo, DecisionRequest, DecisionResponse

_state: dict = {}


def load_calibration(model_id: str) -> TemperatureScaler | None:
    """Load JQV_TEMPERATURE_FILE and verify its provenance against the model / prompt we are about to run.
    Done before the (slow) model load so a mismatch fails fast."""
    path = os.environ.get("JQV_TEMPERATURE_FILE")
    if not path:
        return None
    ts = TemperatureScaler.load(path)
    ts.check_compatible(model_id, prompt_hash(default_style_for(model_id, os.environ.get("JQV_LAYOUT") or None)))
    return ts


@asynccontextmanager
async def lifespan(app: FastAPI):
    model_id = os.environ.get("JQV_MODEL") or DEFAULT_MODEL
    ts = load_calibration(model_id)
    rt = load_runtime(model_id, os.environ.get("JQV_DEVICE"), os.environ.get("JQV_DTYPE"), layout=os.environ.get("JQV_LAYOUT") or None)
    _state["rt"] = rt
    _state["calibration"] = CalibrationInfo(**ts.info()) if ts else None
    kw = {"perm_avg": os.environ.get("JQV_PERM_AVG", "") == "1"}
    if os.environ.get("JQV_HEAD_DIR"):
        kw["head_dir"] = os.environ["JQV_HEAD_DIR"]
    _state["engine"] = make_engine(os.environ.get("JQV_ENGINE", "packed"), rt, ts.temperature if ts else None, **kw)
    yield
    _state.clear()


app = FastAPI(title="jqv Decision API", lifespan=lifespan)


@app.get("/health")
def health():
    rt = _state.get("rt")
    return {"ok": rt is not None, "model": getattr(rt, "model_id", None),
            "engine": getattr(_state.get("engine"), "name", None), "device": str(getattr(rt, "device", None)),
            "prompt_hash": rt.prompt.hash if rt else None,
            "layout": rt.prompt.style.layout if rt else None,
            "calibration": _state["calibration"].model_dump() if _state.get("calibration") else None}


@app.post("/decision", response_model=DecisionResponse)
def decision(req: DecisionRequest) -> DecisionResponse:
    eng = _state["engine"]
    return DecisionResponse(
        engine=eng.name,
        model=_state["rt"].model_id,
        prompt_hash=_state["rt"].prompt.hash,
        temperature=eng.temperature or 1.0,
        calibration=_state.get("calibration"),
        decisions=eng.decide(req.state, req.questions),
    )


@app.post("/v1/systemone")
async def systemone(request: Request):
    """TypeSafe-compatible endpoint (JevBench `typesafe` adapter). Auth headers are accepted and ignored."""
    try:
        body = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="body must be JSON")
    try:
        return decide_systemone(_state["engine"], body, _state["rt"].model_id)
    except SystemOneError as e:
        raise HTTPException(status_code=400, detail=str(e))


def main() -> None:
    import argparse

    import uvicorn

    ap = argparse.ArgumentParser()
    ap.add_argument("--model")
    ap.add_argument("--engine", default="packed")
    ap.add_argument("--temperature-file")
    ap.add_argument("--dtype")
    ap.add_argument("--device")
    ap.add_argument("--perm-avg", action="store_true")
    ap.add_argument("--head-dir")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    a = ap.parse_args()
    for k, v in {"JQV_MODEL": a.model, "JQV_ENGINE": a.engine, "JQV_TEMPERATURE_FILE": a.temperature_file,
                 "JQV_DTYPE": a.dtype, "JQV_DEVICE": a.device, "JQV_PERM_AVG": "1" if a.perm_avg else None,
                 "JQV_HEAD_DIR": a.head_dir}.items():
        if v:
            os.environ[k] = v
    uvicorn.run("jqv.server:app", host=a.host, port=a.port)


if __name__ == "__main__":
    main()
