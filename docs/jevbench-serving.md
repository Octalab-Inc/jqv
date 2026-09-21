# Running jqv yourself (for JevBench / Benchmark Heaven)

jqv is a Decision API on a stock decoder LLM: the state is prefilled once, every question runs as an isolated
branch (block attention mask, no cross-question leakage), and the answer is read directly from the option-letter
logits in one forward pass, with no decoding. It serves TypeSafe's wire format (`POST /v1/systemone`), so the
JevBench `typesafe` adapter works unchanged. This page gives the exact configuration that Benchmark Heaven measured
as a partial row in JevBench v1.2.7 ([issue #6](https://github.com/fstandhartinger/jevbench/issues/6)) and how to
run it on your own hardware. The full experimental report is in [report.md](report.md).

## Measured configuration

| item | value |
|---|---|
| base model | `Qwen/Qwen3-32B` (public weights, Apache-2.0), BF16, no fine-tuning |
| engine | `packed` (one forward per state with a block mask over the question branches) |
| prompt | chat template with thinking disabled, answer read at `Answer:` from the tokens ` A`, ` B`, ...; `prompt_hash` `4f85a0b34776` |
| calibration | one scalar temperature `3.0225814579771493`, fitted on 400 MMLU validation items, stored with provenance in `results/mmlu_packed_qwen3-32b_temperature.json` |
| served probabilities | temperature-scaled distribution over the exact label set (`probabilities` in the response) |
| hardware we used | Apple M5 Max 128 GB, PyTorch MPS, one request at a time |

Public-tier numbers we measured with the JevBench harness at commit `7ce310c` (reproduced exactly by Benchmark
Heaven): easy 1.000, standard 0.958, hard 0.622; hard-tier top-label ECE 0.107.

## Requirements

- Python 3.12 and [uv](https://docs.astral.sh/uv/). `uv sync` installs the locked dependencies (`uv.lock`).
- PyTorch 2.4+ with a GPU backend: Apple MPS (what we used) or CUDA. Memory for the weights: about 65 GB for
  Qwen3-32B in BF16 (28 GB for Qwen3-14B, 3.4 GB for Qwen3-1.7B).
- The weights are downloaded from the Hugging Face Hub on first start (`HF_TOKEN` is optional; it only raises the
  rate limit).

## Start the server

```sh
git clone https://github.com/Octalab-Inc/jqv && cd jqv
uv sync
uv run pytest tests/test_prompt.py tests/test_calibration.py  # checks that need no model download (the full suite uses Qwen3-1.7B)

JQV_MODEL=Qwen/Qwen3-32B JQV_ENGINE=packed \
JQV_TEMPERATURE_FILE=results/mmlu_packed_qwen3-32b_temperature.json \
uv run uvicorn jqv.server:app --host 127.0.0.1 --port 8000
```

`JQV_DEVICE` selects the device (`mps`, `cuda`, `cpu`; default: MPS or CUDA when available) and `JQV_DTYPE` the
dtype (default `bfloat16` on a GPU). Loading the 32B weights takes a few minutes; `GET /health` reports the
configuration once the model is up:

```sh
curl -s localhost:8000/health
# {"ok": true, "model": "Qwen/Qwen3-32B", "engine": "packed", "prompt_hash": "4f85a0b34776",
#  "calibration": {"temperature": 3.0225814579771493, "model": "Qwen/Qwen3-32B", "dataset": "mmlu", "n_val": 400, ...}}
```

The server refuses to start if the temperature file does not match the model and prompt (provenance check); set
`JQV_ALLOW_CALIBRATION_MISMATCH=1` only for experiments.

## Wire format

`POST /v1/systemone` accepts TypeSafe's shape: `state` (a string, or a JSON object/array, which is rendered as
pretty-printed JSON text), and `questions` as an object of `choice` / `noul` / `score` questions. No API key is
needed (`--key-env ''` in the harness); any `model` value is accepted and echoed back.

```sh
curl -s -X POST localhost:8000/v1/systemone -H 'Content-Type: application/json' -d '{
  "model": "jqv-qwen3-32b",
  "state": "The bridge deck shows map cracking with efflorescence.",
  "questions": {
    "damaged":  {"type": "noul",   "instructions": "Is the deck damaged?", "criteria": {"true": "", "false": ""}},
    "severity": {"type": "choice", "instructions": "Pick the severity.",  "criteria": {"minor": "", "moderate": "", "severe": ""}},
    "count":    {"type": "score",  "instructions": "How many defects?",  "criteria": ["none", "one", "two or more"]}
  }}'
# {"answers": {"damaged": {"type": "noul", "noul": 0.98},
#              "severity": {"type": "choice", "choice": "minor", "probabilities": {"minor": 0.53, "moderate": 0.42, "severe": 0.05}},
#              "count": {"type": "score", "score": 1.1, "probabilities": {"0": 0.2, "1": 0.5, "2": 0.3}}},
#  "usage": {"input_tokens": 120, "output_tokens": 0}, "model": "jqv-qwen3-32b"}
```

`POST /decision` is the native endpoint (state + a list of `{question, choices}`); it returns raw and calibrated
probabilities plus a Jev-style confidence.

## Reproduce the public-tier numbers with the JevBench harness

`scripts/jevbench_run.py` starts the server itself on port 8010, runs the harness (`typesafe` adapter, one request
at a time) over the public tiers, and writes `results/jevbench/<label>/<tier>/`:

```sh
git clone https://github.com/fstandhartinger/jevbench /path/to/jevbench   # we used commit 7ce310c
PYTHONPATH=/path/to/jevbench uv run python scripts/jevbench_run.py --model Qwen/Qwen3-32B --engine packed \
  --temperature-file results/mmlu_packed_qwen3-32b_temperature.json --label qwen3-32b_packed_T
uv run python scripts/jevbench_summary.py          # accuracy / ECE table
uv run python scripts/jevbench_families.py         # hard tier by family
```

## Notes on other hardware

- The `packed` engine builds a 4D additive attention mask and runs through PyTorch SDPA; it is exercised on MPS in
  our tests. On CUDA it should work unchanged (`JQV_DEVICE=cuda`), but we have not run it there. If the masked path
  is a problem on your stack, `JQV_ENGINE=kvcache` (shared KV cache, batched questions) and `JQV_ENGINE=naive` (one
  question per forward) produce the same choice logits; `tests/test_engines_equivalence.py` checks that (exact in
  FP32, within BF16 rounding otherwise).
- BF16 logits differ across hardware by rounding only; on our machine the served argmax matched the FP32 result on
  every checked item.
- Latency: on the M5 Max the raw single-decision p50 was 0.65 s on the public standard tier. A CUDA GPU will be faster.
- Other options: `JQV_PERM_AVG=1` averages over the cyclic rotations of the option order (+2 pt on the hard tier
  in our runs, at 2x the latency); `JQV_HEAD_DIR` loads a trained slot head + LoRA (not part of the measured
  configuration). Backbones `Qwen/Qwen3-14B` and `Qwen/Qwen3-1.7B` work with their own temperature files in
  `results/`.
