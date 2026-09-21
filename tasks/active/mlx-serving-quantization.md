---
title: MLX backend（prefill → shared KV → Metal D3 → selected rows readout）と量子化 32B の serving optimization で、この Mac の accuracy / speed Pareto 点を決める
status: draft
priority: P2
created_at: 2026-09-21T10:27:55+09:00
depends_on:
  - metal-shared-attention-d3
---
# Goal

Metal D3 kernel の効果が確認できた後に、jqv に MLX 推論 backend（decision 経路だけ: state の prefill → shared KV → Metal D3 → selected rows readout）を足し、
`JQV_BACKEND=mlx` で `/decision` と `/v1/systemone` を出す。そのうえで 32B bf16 / 8 bit / 4 bit と 14B bf16 を JevBench public hard、MMLU、校正、q/s、メモリ、
単一決定の p50 で比較し、この Mac でのローカル serving の accuracy / speed Pareto 点（例: 32B 8 bit が 14B bf16 を両面で上回るか）を決める。
architecture 研究とは分けて serving optimization として扱う。

# Context

- `metal-shared-attention-d3` の kernel と 14B end-to-end 経路が前提。学習は MLX に移さない。
- Hub に `mlx-community/Qwen3-32B-{4bit,6bit,8bit,bf16}` がある（4 bit ≈ 18 GB、8 bit ≈ 35 GB の見込み）。
- jevmlx（[bnsd55/jevmlx](https://github.com/bnsd55/jevmlx)、README 2026-09-21 確認）: MLX 上で context + schema を 1 回 prefill し、broadcast KV cache に対して
  option を trie 行として restricted softmax で採点。`jevmlx serve` が `/decide` と `/v1/systemone` を出す。既定 alias は Qwen2.5 の 4 bit（7B / 3B / 1.5B）、custom Hub ID 可。
  TypeSafe の private set で 0.4 s/件（one Metal GPU）と記載。prompt と採点方式が違うので同一機の latency だけを比較する。
- Benchmark Heaven の測定: p50 0.92 s（ドイツから）、ローカル 0.65 s、Jev 本番 0.65 s。
- 量子化の効き方: 短い state は帯域律速で 4 / 8 bit が効き、長い prefill は計算律速で効きが小さい。量子化で logits が変わるので温度は model ごとに val 400 で再学習し provenance を残す。
- PyTorch 32B bf16 の現状: JevBench hard 0.622、MMLU 0.809、ECE+T 0.023。

# Scope

## In

- `jqv/engine/mlx/`: mlx-lm の Qwen3（bf16 / 8 bit / 4 bit）を読み、state prefill → 層ごとの shared KV → Metal D3（前タスク）で branch を処理し、
  selected rows readout（`legal_mass` を出せる全語彙 softmax を保持）。Metal kernel が使えない環境向けに加算マスクの packed fallback。
- サーバ: `JQV_BACKEND=mlx` と量子化の指定。温度ファイルは model ごと。
- 等価性: MLX bf16 と PyTorch bf16 の choice logits が許容差内（MMLU 50 問）。量子化は MMLU test 800 と JevBench public hard 111 の精度・ECE を bf16 と比較。
- ベンチ: S ∈ {500, 2000, 8000} × Q ∈ {1, 10, 100} で PyTorch packed / D3 と MLX D3 × {bf16, 8 bit, 4 bit}、メモリ、単一決定（S ≈ 1k、Q=1）の `/v1/systemone` end-to-end p50 / p95（ローカル 50 リクエスト）。
- jevmlx を同一機で起動し（`mlx-community/Qwen3-32B-4bit` が読めなければ既定の Qwen2.5-7B-4bit）、同じ 50 リクエストで latency を測る。
- `/v1/systemone` 経由でハーネスの easy tier を実行して形式互換と精度を確認。
- README「serving optimization」節: {32B bf16, 8 bit, 4 bit, 14B bf16} × {hard, MMLU, ECE+T, q/s, メモリ, p50} の表と推奨構成。

## Out

- MLX 上の学習。CUDA。prompt / readout の変更。温度以外の量子化向け校正。

# Success

- MLX backend 経由のハーネス easy tier が 100%。
- 32B 8 bit の MMLU test が bf16 の ±0.5 pt 以内、JevBench hard が ±2 pt 以内（候補）。4 bit も表に載る。
- 上記の表が `results/mlx_serving.md` にあり、単一決定 p50 の最良構成が 0.35 s 以下（候補。届かない場合は理由を記録）。
- jevmlx の同一機 latency が同じ条件で表にある。README に Pareto 点の結論と推奨構成がある。
- ベンチと評価は進捗行と ETA を出し、途中結果を逐次保存する。

# Verify

```bash
uv run pytest tests/test_mlx_backend.py                                                   # 候補
uv run python scripts/eval.py --dataset mmlu --backend mlx --model mlx-community/Qwen3-32B-8bit --engine shared   # --backend は候補
uv run python scripts/bench.py --backend mlx --engines shared --quant 8bit --state-tokens 500 2000 8000 --questions 1 10 100
JQV_BACKEND=mlx JQV_MODEL=mlx-community/Qwen3-32B-8bit uv run uvicorn jqv.server:app --port 8000 &
PYTHONPATH=/Users/h.imura/tmp/repo/jevbench uv run python scripts/jevbench_run.py --tier easy ...
uvx jevmlx serve --model mlx-community/Qwen3-32B-4bit                                     # 候補。モデル対応を確認
```

# Open Questions

- 既定で serve する量子化（8 bit が本命）。6 bit を含めるか。
