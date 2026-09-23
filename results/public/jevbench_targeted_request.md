# JevBench bench request: jqv-targeted (https://github.com/fstandhartinger/jevbench/issues/51)

Posted 2026-09-23 20:46 JST from the user's GitHub account.

Following up on #9 (jqv, the zero-shot Qwen3-32B row): we would like to request a measurement of a second, separate configuration, `jqv-targeted`, and to keep the existing `jqv` row as it is.

**What it is.** The same server, base weights (Qwen/Qwen3-32B, BF16), prompt (`prompt_hash 4f85a0b34776`) and wire format as the ranked row, plus a trained readout: a LoRA adapter (r=16 on q/k/v/o_proj, 40 M parameters) and a small "slot" head, trained for 600 steps × batch 8 on program-verified synthetic decision items (long policies, temporal/numeric rules, probability; answers produced by solvers, not by a model) mixed with MMLU. Nothing is generated at inference; the answer is still read from one forward pass, so latency is the zero-shot row's within noise.

**Please note, for the exposure question.** No JevBench item is in the training data (0 shared word 8-grams between the synthetic sets and the public items, checked with `scripts/synth_contamination.py`). But the synthetic generators were designed after reading which public hard items the zero-shot model and a first targeted head got wrong (the temporal_numeric family), so for us the public hard tier is a development gate, not a held-out measurement. That is exactly why we would like it measured on your side: the held-out and sealed items are the real test of whether this kind of targeted post-training generalises. If your rules place such a row in a particular class or with a note, that is fine with us.

**How to run it.** Code: https://github.com/Octalab-Inc/jqv at `main` (commit `c1bcf11`). Checkpoint: release [`targeted-v2-32b`](https://github.com/Octalab-Inc/jqv/releases/tag/targeted-v2-32b), `jqv-targeted-v2-32b.tar.gz` (148 MB), SHA-256 `fbdcb2da42882775e28acb5185d927a8d71107ec5c26fa8216a3718f062e6303`.

```sh
git clone https://github.com/Octalab-Inc/jqv && cd jqv && uv sync
curl -L -o jqv-targeted-v2-32b.tar.gz https://github.com/Octalab-Inc/jqv/releases/download/targeted-v2-32b/jqv-targeted-v2-32b.tar.gz
shasum -a 256 jqv-targeted-v2-32b.tar.gz && tar -xzf jqv-targeted-v2-32b.tar.gz
JQV_MODEL=Qwen/Qwen3-32B JQV_ENGINE=slot JQV_HEAD_DIR=jqv-targeted-v2-32b/best \
JQV_TEMPERATURE_FILE=jqv-targeted-v2-32b/temperature.json uv run uvicorn jqv.server:app --host 127.0.0.1 --port 8000
```

`GET /health` should show `"engine": "slot"`, `"prompt_hash": "4f85a0b34776"` and a calibration block with `"temperature": 1.818988859597101`, `"dataset": "mmlu"`, `"n_val": 400` (the server refuses a temperature file that does not match the model and prompt). Full instructions, including the harness invocation, are in [docs/jevbench-serving.md](https://github.com/Octalab-Inc/jqv/blob/main/docs/jevbench-serving.md#targeted-configuration-jqv-targeted-the-same-server-with-a-trained-slot-head--lora). We ran it on an NVIDIA GB10 (CUDA 13, torch 2.14); the engine is the same masked-attention path you ran on the H100 for #9, with the LoRA applied through PEFT.

**What we measured on the public tiers** (harness commit 7ce310c, one request at a time): easy 1.000, standard 0.958, hard 82 / 111 = 0.739, hard-tier ECE 0.118, Brier 0.390. On the same 111 hard items the zero-shot row is 68 / 111 (paired: 19 won / 5 lost, exact McNemar p = 0.007); by family the largest moves are temporal_numeric 4 → 7 / 15 and probability 4 → 9 / 10. The report with the training design, the 14B gate and the caveats is in [docs/report.md](https://github.com/Octalab-Inc/jqv/blob/main/docs/report.md).

Thanks again for running these; we will not make any claim about held-out or sealed items until your run.
