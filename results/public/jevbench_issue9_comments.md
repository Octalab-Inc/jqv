## fstandhartinger — 2026-09-21T03:12:45Z
https://github.com/fstandhartinger/jevbench/issues/9#issuecomment-5754892987

Thanks — this is logged for reachability checks and the JevBench measurement queue. We’ll only make benchmark claims after our own run.

- Harold

## fstandhartinger — 2026-09-21T06:17:13Z
https://github.com/fstandhartinger/jevbench/issues/9#issuecomment-5756199481

Re-run done on our own hardware from `Octalab-Inc/jqv` 0189b67: `/health` showed prompt_hash 4f85a0b34776 and
temperature 3.0225814579771493 before the run, Qwen/Qwen3-32B BF16, packed engine, on a RunPod H100 NVL (torch 2.11 +
CUDA 12.8), one request at a time. This time all 534 decisions went out, including the 109 held-out hard items, so the
v1.2.7 partial row is replaced by a complete, ranked one.

**#8 of 36, JevBench Score 70.1** (Intelligence 86.1, Calibration 79.0, Speed 74.6, Cost 47.5). Accuracy: easy 100.0 %, standard 95.8 %, judge 92.5 %, hard 64.5 %. Latency p50 0.75 s raw, 1.64 s after the standard self-host adjustment (x2 + 0.15 s); our pod was in Canada, so the raw figure includes the trip from our server in Germany; cost $0.0564 per 1,000 decisions.

The public tiers came out as in #6 (easy 100 %, standard/judge unchanged); on the full hard tier it answered
64.5 %. The earlier tunnel run stays in the history of the repo. Thanks for publishing the serving code.

Page: https://benchmarkheaven.com/jev-models · data and method: https://github.com/fstandhartinger/jevbench/tree/v1.2.8

- Harold


