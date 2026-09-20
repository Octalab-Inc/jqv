Request to add **jqv** (Qwen3-32B, Apache-2.0 base weights, no fine-tuning; TypeSafe wire format) to the ranked systems, via a public endpoint.

**What it is.** An open reconstruction of the Jev inference design on a stock decoder LLM: the state is prefilled once, every question runs as an isolated branch (block attention mask, no cross-question leakage), and the answer is read directly from the option-letter logits in one forward pass (no decoding). The served probabilities are temperature-scaled with a single scalar (T = 3.02) fitted on 400 MMLU validation items; provenance is reported by `GET /health`. The submitted configuration is the zero-shot backbone without any decision training. It serves TypeSafe's wire format (`POST /v1/systemone`, choice / noul / score, string or JSON state), so your `typesafe` adapter works unchanged.

**Endpoint (self-hosted, no billing).**

- `https://asn-front-mix-develop.trycloudflare.com` — a Cloudflare Quick Tunnel to an Apple M5 Max (128 GB) in Japan running Qwen3-32B in BF16 on PyTorch/MPS, one request at a time.
- No API key: run with `--key-env ''`; any `model` value is accepted and echoed back (we use `jqv-qwen3-32b`).
- Cost basis: `no_billable_account_public_endpoint`.
- Please send one warm-up request before timing. Raw latency on the machine is p50 0.65 s on the public standard tier; your wall time will include the network path from Germany to Japan.
- The tunnel stays up until your run is done — please comment here when finished so I can take it down. The source is not public yet; I can share it on request.

```sh
python -m jevbench.cli run --tasks datasets/public/original.jsonl \
  --adapter typesafe --endpoint https://asn-front-mix-develop.trycloudflare.com --key-env '' --model jqv-qwen3-32b \
  --cost-basis no_billable_account_public_endpoint --reserve-usd 0 --delay-s 0.2 ...
```

**Our own numbers on the public items** (your harness at commit 7ce310c, `typesafe` adapter, one request at a time, argmax over the exact label set, served probabilities as above):

| tier | items | jqv (Qwen3-32B, zero-shot) |
|---|---|---|
| easy | 48 | 1.000 |
| standard (original) | 72 | 0.958 |
| hard | 111 | 0.622 |

Hard-tier top-label ECE 0.107 (raw p50 0.65 s). No distribution-gold items are public, so we have no distribution-fidelity number. We are not claiming a rank from this: the held-out and judge items are yours, and speed is measured from your server.

**Disclosure.** No training was done on any benchmark item; the only fitted quantity is the temperature (MMLU validation). The public 231 items were used to evaluate nine configurations (three backbone sizes × zero-shot / option-order averaging / LoRA head), and the choice to submit the zero-shot 32B configuration rather than the option-averaged one was made after seeing those public-tier numbers, so treat the public tiers as a development gate once. JSON-state items are rendered to the model as pretty-printed JSON text.
