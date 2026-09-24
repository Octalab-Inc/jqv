# jqv — a Decision API on Qwen3 (a Jev-style reconstruction)

jqv reproduces, on a stock open LLM (Qwen3), the inference structure of TypeSafe's Jev as reconstructed by
[Hume](https://archerhume.com/posts/jevs-architecture-unmasked/?v=3): the state is prefilled once, every question runs as an
isolated branch behind a block attention mask, and the answer is read directly from the option-letter logits in one forward
pass, with no decoding. A single fitted temperature turns the readout into calibrated probabilities. The server speaks
TypeSafe's wire format (`POST /v1/systemone`) as well as its own `POST /decision`.

- Full experimental report: [docs/report.md](docs/report.md) (Japanese original: [docs/report.ja.md](docs/report.ja.md))
- Running the JevBench-measured configuration yourself: [docs/jevbench-serving.md](docs/jevbench-serving.md)
- License: Apache-2.0

## What it does

```
POST /decision
{"state": "Bridge A: extensive corrosion on the lower flange of the main girder ...",
 "questions": [{"question": "What damage does the main girder show?", "choices": ["none", "corrosion", "cracking"]}, ...]}
→ {"decisions": [{"probabilities": [0.01, 0.97, 0.02], "calibrated_probabilities": [...], "confidence": 0.9}, ...]}
```

Five inference structures and two readouts can be switched on the same model and prompt:

| engine | structure |
|---|---|
| `generate` | ordinary generation, parse the emitted letter (baseline) |
| `naive` | one forward per question, read the option-letter logits at the last position |
| `kvcache` | prefill the state once, replicate the KV cache, batch the questions |
| `packed` | `[state \| q1 \| q2 \| ...]` in one sequence with a block attention mask (reference implementation) |
| `shared` | the same input through a mask-free Hydragen-style attention: branch queries attend to the shared prefix together, branch-local causal attention, log-sum-exp combination |
| `readout="rows"` | compute the logits from the option-letter rows of the LM head only (no full-vocabulary projection) |

In fp32 the choice logits of `naive` / `kvcache` / `packed` / `shared` agree (tests), sibling questions cannot see each other
(leakage 0.000 against a negative control of 0.996), and at S=8k tokens × 100 questions the shared engines are 53-73x faster
than one forward per question on an Apple M5 Max.

## Results in brief

| backbone (zero-shot, packed) | MMLU test 800 | JMMLU test 800 | MMLU ECE raw / +T | JevBench public hard (111) |
|---|---:|---:|---|---:|
| Qwen3-1.7B | 0.554 | 0.466 | 0.413 / 0.080 | 0.423 |
| Qwen3-14B | 0.750 | 0.710 | 0.207 / 0.042 | 0.550 |
| Qwen3-32B | 0.809 | 0.771 | 0.137 / 0.023 | 0.622 |
| Jev (TypeSafe; Hume / Benchmark Heaven) | 0.918 | - | 0.031 | 0.741 |

Benchmark Heaven measured the 32B zero-shot configuration twice: first through a tunnel to our machine
([JevBench v1.2.7](https://github.com/fstandhartinger/jevbench/blob/v1.2.7/RESULTS-v1.2.md), a partial row because the
endpoint was submitter-operated) and then, after this code was published, on their own RunPod H100 from commit 0189b67
([JevBench v1.2.8](https://github.com/fstandhartinger/jevbench/tree/v1.2.8), [jevbench#9](https://github.com/fstandhartinger/jevbench/issues/9)).
The full run is a ranked row: **#8 of 36, JevBench Score 70.1** (Intelligence 86.1, Calibration 79.0, Speed 74.6, Cost 47.5);
under the later re-scorings of the same measurements it is **#6 of 48 (68.6)** in v1.3.0 (chance-corrected Intelligence) and
**#12 of 71 (44.4)** in v1.4.0, which adds 308 sealed decisions on which jqv scores 0.282 (field median 0.292, Jev 1.13.0 0.367);
easy 1.000, standard 0.958, judge 0.925, hard 0.645 on all 220 items (Jev 1.13: 0.741); p50 0.75 s raw from Germany to a pod in
Canada; $0.0564 per 1,000 decisions at the base model's public tariff.

Findings, in one line each (details in the report):

1. Jev's main inference behaviours (direct readout, shared state, sibling isolation, listwise option interaction) are
   reproducible on an ordinary open decoder.
2. The speed-up comes from sharing the state, not from "not generating": generate ≈ naive, shared engines 53-73x at
   long states.
3. Accuracy is set mainly by backbone scale; a slot head + LoRA trained on 4,800 examples helps at 1.7B and not at 14B/32B,
   and the gap to Jev stays at about 10 points on MMLU (10.9) and JevBench hard (9.6) alike.
4. A scalar temperature reaches Jev's reported ECE in and near the fitting distribution (MMLU 0.023), transfers across
   languages (MMLU ↔ JMMLU), transfers partially to JevBench hard and is harmful on a reading-type task.
5. Option-order averaging (`perm_avg`) is a training-free +3 points at 1.7B/14B and cancels the position and letter priors.
6. Targeted LoRA on program-verified synthetic data for the weak JevBench families is selective transfer, not a general lift:
   +22 to +42 points in-distribution at 14B/32B, but JevBench hard moves 68 → 72 / 111 at 32B (13 won / 9 lost, p=0.52) with
   better calibration (ECE 0.127 → 0.096, Brier 0.516 → 0.415); probability transfers, long-policy barely, and temporal-numeric
   training hurts reproducibly (5 → 3 / 15 at 14B, 4 → 2 at 32B).
7. Rebuilding the temporal generator around the computations those items need (term vs cap, FX lines with a per-night cap,
   earliest-of expiry conditions, deadline booleans, AND/OR tiers) removes the negative transfer at 14B (temporal 3 → 6 / 15) and
   lifts the public hard tier to 74 / 111 (22 won / 9 lost vs zero-shot, p=0.029), above the 32B zero-shot; the two items the
   scenarios were modelled on are still missed, so the gain is computation-type generalisation, not memorisation.
8. The same v2 mixture at 32B reaches 82 / 111 on the public hard tier (19 won / 5 lost vs zero-shot, p=0.007; temporal 7 / 15,
   probability 9 / 10, Brier 0.390) with MMLU/JMMLU unchanged; ECE (0.118) is worse than the first targeted head (0.096). This
   configuration is submitted to Benchmark Heaven as its own row, `jqv-targeted` (checkpoint in the `targeted-v2-32b` release,
   instructions in [docs/jevbench-serving.md](docs/jevbench-serving.md)); the held-out and sealed items are the real test.
9. Prompt-order ablation at 32B: reading the question before the state (Q → state → Q) moves JevBench hard by +3 / 111 (7 won /
   4 lost, p=0.55; gains on multi_hop, temporal and probability, losses on ambiguous) while costing 7.5× at 10 questions per state
   and 18× at 100, so the state-first shared prefill gives up almost nothing; repeating the question after the state adds +1.9 MMLU
   points but nothing on JevBench hard and loses 4 standard items.

## Quick start

```bash
uv sync                                   # Python 3.12, PyTorch (MPS or CUDA), transformers 5.x
uv run pytest                             # downloads Qwen/Qwen3-1.7B on first run

# the JevBench-measured configuration (Qwen3-32B, ~65 GB in bf16; use Qwen/Qwen3-1.7B to try it quickly)
JQV_MODEL=Qwen/Qwen3-32B JQV_ENGINE=packed \
JQV_TEMPERATURE_FILE=results/mmlu_packed_qwen3-32b_temperature.json \
uv run uvicorn jqv.server:app --host 127.0.0.1 --port 8000

curl -s localhost:8000/health             # model, engine, prompt_hash, calibration provenance
curl -s -X POST localhost:8000/v1/systemone -H 'Content-Type: application/json' -d '{
  "model": "jqv", "state": "The bridge deck shows map cracking with efflorescence.",
  "questions": {"damaged": {"type": "noul", "instructions": "Is the deck damaged?", "criteria": {"true": "", "false": ""}},
                "severity": {"type": "choice", "instructions": "Pick the severity.", "criteria": {"minor": "", "moderate": "", "severe": ""}}}}'
```

```python
from jqv.model import load_runtime
from jqv.engine import make_engine
from jqv.types import Question
rt = load_runtime("Qwen/Qwen3-1.7B")
eng = make_engine("packed", rt, temperature=12.0)
eng.decide(state, [Question(question="...", choices=["...", "..."])])
```

The server refuses a temperature file whose model or prompt hash does not match the runtime (provenance check).

## Experiments

```bash
uv run scripts/eval.py --dataset mmlu --engine packed --n 1200 --n-val 400     # accuracy / NLL / Brier / ECE, logits cached
uv run scripts/fit_temperature.py results/mmlu_packed_qwen3-1.7b.npz           # fit T on val, before/after on test
uv run scripts/transfer_temperature.py                                          # T transfer across datasets
uv run scripts/bench.py --state-tokens 500 2000 8000 --questions 1 10 100      # latency per engine, resumable
uv run scripts/isolation_test.py                                               # secret-code isolation with a negative control
uv run scripts/permutation_test.py --mode label|order|fifth                     # letter prior, order prior, fifth option
uv run scripts/train_head.py --run-name r1 --head slot --lora-rank 16           # slot / pointer heads, CE + λ·Brier
uv run scripts/compare_runs.py ...                                              # paired McNemar, bootstrap CI, selective accuracy
PYTHONPATH=/path/to/jevbench uv run scripts/jevbench_run.py --model Qwen/Qwen3-32B --engine packed \
  --temperature-file results/mmlu_packed_qwen3-32b_temperature.json --label qwen3-32b_packed_T   # JevBench public tiers
```

Results (JSON / Markdown / PNG) are under `results/`; the JevBench runs are under `results/jevbench/<label>/`.

## Sample application: jqgrep

[`jqgrep/`](jqgrep/README.md) is a cascade semantic code search built on jqv with no index and no embeddings: a lexical
sketch stage, grouped shared-state routing with a listwise question, then one forward per candidate file that scores every
line range and the file's role (implementation / call site / test / config) against the query. With Qwen3-14B the three
reference queries in its README each return the expected file as the sole top hit.

```sh
uv run python -m jqgrep "where is the block attention mask for packed sequences built" . --model Qwen/Qwen3-14B
```

## Layout

```
jqv/            prompt.py (prefix/suffix, prompt_hash), readout.py, engine/ (generate, naive, kvcache, packed, shared, head),
                calibration.py (temperature with provenance), heads.py + train/ (slot / pointer heads, LoRA, CE + λ·Brier),
                server.py (/decision, /v1/systemone, /health), systemone.py (TypeSafe wire format), data.py, synth/ (synthetic hard-family data)
scripts/        evaluation, calibration, benchmarks, isolation and permutation tests, training, JevBench harness driver
tests/          prompt equivalence, engine equivalence (fp32 exact / bf16 tolerance), isolation, calibration, heads, systemone
results/        cached logits, temperature files, benchmark tables, JevBench runs and the published v1.2.7 row
docs/           report.md (full report), report.ja.md, jevbench-serving.md
jqgrep/         sample application: cascade semantic code search on jqv
```

## Related projects

Independent 2026 implementations of the same ideas include
[featherless-ai/simple-jev](https://github.com/featherless-ai/simple-jev), [TianyuCodings/NanoJev](https://github.com/TianyuCodings/NanoJev),
[TheoLeeCJ/SemIf](https://github.com/TheoLeeCJ/SemIf), [ekzhang/openjev-sglang](https://github.com/ekzhang/openjev-sglang),
[r-ms/mini-jev](https://github.com/r-ms/mini-jev), [zwliJay/jev-forge](https://github.com/zwliJay/jev-forge) and
[bnsd55/jevmlx](https://github.com/bnsd55/jevmlx); the shared-prefix attention follows [Hydragen](https://arxiv.org/abs/2402.05099).
What jqv adds is the engine decomposition with equivalence tests, the isolation negative control, and the calibration and
temperature-transfer measurements; see the report for the comparison.

## License

Apache-2.0. The Qwen3 weights are Apache-2.0 as well.
