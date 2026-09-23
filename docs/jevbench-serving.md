# Running jqv yourself (for JevBench / Benchmark Heaven)

jqv is a Decision API on a stock decoder LLM: the state is prefilled once, every question runs as an isolated
branch (block attention mask, no cross-question leakage), and the answer is read directly from the option-letter
logits in one forward pass, with no decoding. It serves TypeSafe's wire format (`POST /v1/systemone`), so the
JevBench `typesafe` adapter works unchanged. This page gives the exact configuration that Benchmark Heaven measured
as a partial row in JevBench v1.2.7 ([issue #6](https://github.com/fstandhartinger/jevbench/issues/6)) and then, from
this repository at commit 0189b67 on their own RunPod H100 NVL (torch 2.11 + CUDA 12.8), as a ranked row in v1.2.8
([issue #9](https://github.com/fstandhartinger/jevbench/issues/9): #8 of 36, JevBench Score 70.1), and how to run it
on your own hardware. The full experimental report is in [report.md](report.md).

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

## Targeted configuration (`jqv-targeted`): the same server with a trained slot head + LoRA

A second configuration, requested as its own row (`jqv-targeted`), keeps the base model, prompt (`prompt_hash`
`4f85a0b34776`), engine family and wire format above and adds a trained readout: a LoRA adapter (r=16 on q/k/v/o_proj,
40 M parameters) and a slot head, trained for 600 steps × batch 8 with `scripts/train_head.py` on program-verified
synthetic decision items (`data/synth/`: long_policy, temporal_numeric, temporal_v2, probability; generated by
`jqv/synth/`, every answer produced by a solver, 0 shared word 8-grams with the JevBench public items per
`scripts/synth_contamination.py`) mixed with MMLU auxiliary_train (30 %). The generator scenarios were designed after
reading which public hard items the zero-shot model and a first targeted head lost (temporal_numeric), so the public
hard tier below is a development gate for us, not a held-out measurement; the held-out and sealed items are the test.
Details and the 14B gate: [report.md](report.md), sections "Targeted LoRA on the weak families" and "Temporal
generator v2".

| item | value |
|---|---|
| base model | `Qwen/Qwen3-32B`, BF16 (unchanged) |
| engine | `slot`: the `packed` forward plus the LoRA adapter, answer read by the slot head at the same position |
| head | `jqv-targeted-v2-32b/best` (run `32b-hardfam-v2`, step 600); `best/head/head.json` records model and `prompt_hash` and the server refuses a mismatch |
| calibration | one scalar temperature `1.818988859597101`, fitted on 400 MMLU validation items with this head (`temperature.json` in the archive = `results/mmlu_slot_qwen3-32b_32b-hardfam-v2_temperature.json`) |
| checkpoint | [release `targeted-v2-32b`](https://github.com/Octalab-Inc/jqv/releases/tag/targeted-v2-32b): `jqv-targeted-v2-32b.tar.gz` (148 MB), SHA-256 `fbdcb2da42882775e28acb5185d927a8d71107ec5c26fa8216a3718f062e6303` |
| hardware we used | NVIDIA GB10 (DGX Spark class, CUDA 13.0, torch 2.14), one request at a time |

```sh
curl -L -o jqv-targeted-v2-32b.tar.gz https://github.com/Octalab-Inc/jqv/releases/download/targeted-v2-32b/jqv-targeted-v2-32b.tar.gz
shasum -a 256 jqv-targeted-v2-32b.tar.gz   # fbdcb2da42882775e28acb5185d927a8d71107ec5c26fa8216a3718f062e6303
tar -xzf jqv-targeted-v2-32b.tar.gz          # -> jqv-targeted-v2-32b/{best/adapter,best/head,config.json,temperature.json,...}

JQV_MODEL=Qwen/Qwen3-32B JQV_ENGINE=slot JQV_HEAD_DIR=jqv-targeted-v2-32b/best \
JQV_TEMPERATURE_FILE=jqv-targeted-v2-32b/temperature.json \
uv run uvicorn jqv.server:app --host 127.0.0.1 --port 8000
```

`GET /health` then reports `"engine": "slot"`, `"prompt_hash": "4f85a0b34776"` and a calibration block with
`"temperature": 1.818988859597101`, `"engine": "slot"`, `"dataset": "mmlu"`, `"n_val": 400`. The harness run is the
same as above with `--engine slot --head-dir jqv-targeted-v2-32b/best --temperature-file jqv-targeted-v2-32b/temperature.json`.

Public-tier numbers we measured with the harness at commit `7ce310c` (hard tier by family in
`results/jevbench/qwen3-32b_hardfam_v2_T/`): hard 82 / 111 = 0.739 (zero-shot configuration above: 68 / 111 on the
same machine; paired 19 won / 5 lost, exact McNemar p = 0.007), hard-tier top-label ECE 0.118, Brier 0.390. The
adapter adds no latency to the forward pass beyond the LoRA matmuls; measured single-decision throughput on the
GB10 was the same as the zero-shot configuration within noise.

## Notes on other hardware

- The `packed` engine builds a 4D additive attention mask and runs through PyTorch SDPA; it is exercised on MPS in
  our tests, and Benchmark Heaven ran it unchanged on an H100 NVL (torch 2.11 + CUDA 12.8) for v1.2.8 with the
  device auto-detected. If the masked path is a problem on your stack, `JQV_ENGINE=kvcache` (shared KV cache, batched questions) and `JQV_ENGINE=naive` (one
  question per forward) produce the same choice logits; `tests/test_engines_equivalence.py` checks that (exact in
  FP32, within BF16 rounding otherwise).
- BF16 logits differ across hardware by rounding only; on our machine the served argmax matched the FP32 result on
  every checked item.
- Latency: on the M5 Max the raw single-decision p50 was 0.65 s on the public standard tier; on the maintainers' H100 it was 0.75 s including the Germany-Canada round trip (hard-tier p50 0.81 s, p95 1.53 s).
- Other options: `JQV_PERM_AVG=1` averages over the cyclic rotations of the option order (+2 pt on the hard tier
  in our runs, at 2x the latency); `JQV_HEAD_DIR` loads a trained slot head + LoRA (not part of the measured
  configuration). Backbones `Qwen/Qwen3-14B` and `Qwen/Qwen3-1.7B` work with their own temperature files in
  `results/`.
