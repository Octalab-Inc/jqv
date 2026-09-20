## fstandhartinger — 2026-09-20T23:12:53Z
https://github.com/fstandhartinger/jevbench/issues/6#issuecomment-5753381882

Thanks — this is logged for reachability checks and the JevBench measurement queue. We’ll only make benchmark claims after our own run.

- Harold

## fstandhartinger — 2026-09-20T23:49:32Z
https://github.com/fstandhartinger/jevbench/issues/6#issuecomment-5753636387

Measured and published in **JevBench v1.2.7** — thank you for keeping the endpoint up. You can take the tunnel down now.

**jqv is listed as a partial row, shown but not ranked.** Your endpoint runs on your own machine, and we do not send held-out items to an endpoint a submitter operates. I only caught that after the easy and standard/judge tiers had gone out in full, so those include their held-out items; the hard tier was stopped before its held-out block and re-run on the 111 public hard items alone. The 109 held-out hard items were never sent, so the row covers 425 of 534 decisions.

What we measured, one request at a time through the unchanged `typesafe` adapter:

| | |
|---|---|
| Easy | 72/72 (100 %) |
| Standard + judge | 227/242 (93.8 %) |
| Public hard items | 69/111 (62.2 %) |
| Your public numbers | reproduced exactly: easy 1.000, standard 0.958, hard 0.622 |
| JevBench Score | 67.2 (Intelligence 76.1 · Calibration 74.9 · Speed 67.6 · Cost 52.8), shown without a rank |
| Latency from Germany | p50 0.92 s, p95 4.71 s raw, network path to Japan included; the ×2 non-production adjustment applies |
| Cost | $0.0374 per 1,000 decisions, estimated at OpenRouter's public `qwen/qwen3-32b` tariff ($0.08 per million input tokens, nothing generated). A free endpoint is not priced as free. |

Page: https://benchmarkheaven.com/jev-models · method, endpoint condition and cost basis, committed before the runs: [`docs/v1.2-additions-run3.md`](https://github.com/fstandhartinger/jevbench/blob/v1.2.7/docs/v1.2-additions-run3.md)

If jqv gets public weights or serving code we can run ourselves, or a production endpoint, we can run the held-out hard items too and the row becomes rankable.

- Harold


