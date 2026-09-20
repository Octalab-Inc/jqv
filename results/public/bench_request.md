# [bench request]: Add jqv (Qwen3-32B, TypeSafe wire format)

- **System**: jqv — Qwen3-32B, one-token vocabulary readout over a shared state (no generation), temperature-scaled probabilities
- **Repository**: /Users/h.imura/tmp/repo/jqv (public URL to be added)
- **Adapter**: `typesafe` (native `POST /v1/systemone`; choice / noul / score; JSON or string state; no auth header needed, `--key-env ''`)
- **Endpoint**: https://asn-front-mix-develop.trycloudflare.com
- **Model id**: `jqv-qwen3-32b` (any value is accepted and echoed back)
- **Weights**: Qwen/Qwen3-32B (Apache-2.0), BF16, Apple M5 Max 128 GB, PyTorch MPS, single request at a time
- **Calibration**: probabilities are temperature-scaled (T=3.02 fitted on 400 MMLU validation items; provenance in `/health`)
- **Cost basis**: self-hosted, no billing (`--cost-basis no_billable_account_public_endpoint`)
- **Public-tier numbers measured locally with your harness** (commit 7ce310c): easy 48/48, standard 69/72 (0.958), hard 69/111 (0.622), hard ECE 0.107, raw p50 0.65 s
- **Note**: the endpoint is a Cloudflare Quick Tunnel from Japan; please expect network latency on top of the raw p50. It stays up until the run is done; ping me when finished so I can take it down.
