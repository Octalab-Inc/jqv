---
title: jqv 32B の TypeSafe 互換 endpoint を Cloudflare Quick Tunnel で一時公開し、JevBench held-out 評価の bench request に備える
status: done
priority: P2
created_at: 2026-09-21T06:21:34+09:00
depends_on: []
---

# Goal

Mac 上の jqv（Qwen3-32B zero-shot、MMLU val の温度 T=3.0）を Cloudflare Quick Tunnel で HTTPS 公開し、JevBench の `typesafe` adapter が
外部から `POST /v1/systemone` を叩ける状態にする。Benchmark Heaven の held-out 303 問 + judge tier の測定（bench request）に使う。

# Context

- `/v1/systemone` は TypeSafe 互換で、サーバに `JQV_TEMPERATURE_FILE` があれば校正済み確率を返す（`jevbench-calibrated-serving` で修正済み）。
- 提出するのは 32B zero-shot（hard 0.622、Calibration 78.6、p50 0.65 s）。perm_avg は +1.8 pt に対し p50 が 2 倍でスコア上不利。
- Quick Tunnel（`cloudflared tunnel --url`）はアカウント不要、ランダム URL、停止で消える。Benchmark Heaven はドイツから concurrency=1 で呼ぶため、
  外部 p50 にはネットワーク遅延が乗る（ランキングでは self-hosted に ×2 + 0.15 s の補正が加わる）。
- 測定中は Mac をスリープさせない（`caffeinate -dimsu`）。公開中は URL を知る誰でも叩けるので、測定後すぐ停止する。

# Scope

## In

- `brew install cloudflared`、`caffeinate`、サーバ起動（port 8000）、ローカル curl で校正済み確率の確認、Quick Tunnel の発行、公開 URL 経由の疎通確認
- README に「公開 endpoint の立て方」を追記（手順と注意点）
- bench request の文面案（endpoint、model id、repo、adapter）を用意する。提出自体はユーザーの確認後

## Out

- 常設 endpoint、認証、Linux/CUDA へのデプロイ

# Success

- 公開 HTTPS URL で `/health` と `/v1/systemone` が応答し、返る確率が校正済み（`/health` の calibration に T=3.0 が出る）
- ローカルと公開 URL で同じ質問に同じ確率が返る
- README に手順が書かれている

# Verify

```bash
curl -s https://<random>.trycloudflare.com/health
curl -s https://<random>.trycloudflare.com/v1/systemone -H 'Content-Type: application/json' -d '{"model":"jqv-qwen3-32b","state":"The payment was charged twice.","questions":{"duplicate":{"type":"noul","instructions":"Was the payment duplicated?"}}}'
```

# Open Questions

- Benchmark Heaven への bench request をいつ出すか（Quick Tunnel は停止すると URL が消えるため、測定が終わるまで維持が必要）

# Result

## Changed

- `README.md`: 「公開 endpoint（Cloudflare Quick Tunnel）」節（手順、注意点、実測遅延、提出設定）
- `results/public/`: `tunnel_url.txt`、`bench_request.md`（提出文面案）、`server.log`、`cloudflared.log`、`harness_*/`（公開 URL 経由のハーネス実行結果）
- 環境: `brew install cloudflared`（2026.9.1）。起動中: `caffeinate -dimsu`、`uvicorn jqv.server:app --port 8000`（32B、T=3.02）、`cloudflared tunnel --url http://localhost:8000`

## Verified

- 公開 URL https://asn-front-mix-develop.trycloudflare.com で `/health`（calibration.temperature=3.02）と `/v1/systemone` が応答。ローカルと同じ質問で同じ確率（bf16 差 0.001）
- 遅延: 同一リクエストでローカル 0.23〜0.74 s、トンネル経由 +0.04〜0.12 s（日本国内から）
- ハーネス `typesafe` adapter を公開 URL に向けて easy 10 問: 10/10、失敗 0、p50 0.30 s、operational_success 1.0

## Deviations

- Benchmark Heaven への bench request は未提出（外部向け行為のためユーザー確認待ち）。Quick Tunnel の URL は `cloudflared` 停止で消えるので、提出から測定完了までサーバとトンネルを維持する必要がある
- README に一度未計測の遅延値を書いてしまい、計測値で置き換えた

## Remaining

- bench request の提出と、測定完了後の停止（`pkill -f "cloudflared tunnel"; pkill -f "uvicorn jqv.server"; pkill caffeinate`）
