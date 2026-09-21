Follow-up to #6 (jqv, partial row in v1.2.7): the serving code is now public, so I would like to request a re-run that can include the held-out hard items on your own hardware, making the row rankable.

**Repository.** https://github.com/Octalab-Inc/jqv — commit `0189b67`. Weights are the public `Qwen/Qwen3-32B` (Apache-2.0, BF16, no fine-tuning). Run instructions written for you: [`docs/jevbench-serving.md`](https://github.com/Octalab-Inc/jqv/blob/0189b67/docs/jevbench-serving.md).

**Configuration.** Identical to the one you measured in #6: `packed` engine (state prefilled once, each question an isolated branch behind a block attention mask, answer read from the option-letter logits in one forward pass, no decoding), one scalar temperature 3.0225814579771493 fitted on 400 MMLU validation items and stored with provenance in the repo (`results/mmlu_packed_qwen3-32b_temperature.json`), served over TypeSafe's wire format (`POST /v1/systemone`), so the `typesafe` adapter runs unchanged with `--key-env ''`. `GET /health` reports `prompt_hash` `4f85a0b34776` and the temperature, which lets you confirm the configuration matches #6 before the run.

```sh
git clone https://github.com/Octalab-Inc/jqv && cd jqv && uv sync
JQV_MODEL=Qwen/Qwen3-32B JQV_ENGINE=packed JQV_TEMPERATURE_FILE=results/mmlu_packed_qwen3-32b_temperature.json \
uv run uvicorn jqv.server:app --host 127.0.0.1 --port 8000
curl -s localhost:8000/health   # model, engine, prompt_hash 4f85a0b34776, calibration.temperature 3.0225814579771493
```

**Hardware.** About 65 GB of memory for the BF16 weights. We ran it on an Apple M5 Max through PyTorch/MPS; on CUDA it should work unchanged (`JQV_DEVICE=cuda`) but we have not run it there. If the masked-attention path gives trouble on your stack, `JQV_ENGINE=kvcache` or `JQV_ENGINE=naive` produce the same choice logits (checked by the equivalence tests in the repo).

**Expected numbers.** On the public tiers the configuration reproduced our own numbers exactly in #6 (easy 1.000, standard 0.958, hard 0.622, hard-tier top-label ECE 0.107), so a healthy run on your side should show the same public-tier figures before the held-out items.

**Cost basis.** Whatever you apply to open weights you run yourselves; the OpenRouter `qwen/qwen3-32b` input tariff you used in #6 is fine with me.

**Disclosure.** Unchanged from #6: no benchmark item was used for training; the only fitted quantity is the temperature (MMLU validation); the public 231 items were used once as a development gate when choosing this configuration over an option-order-averaged variant.

Thank you for the v1.2.7 run and for documenting the endpoint condition; happy to answer questions here.
