---
title: Apple Silicon 上で serving を詰める（MLX 版 packed / shared engine、量子化、jevmlx との同一機比較）で、速度差のうちソフトウェア由来の分を測る
status: draft
priority: P2
created_at: 2026-09-21T10:11:14+09:00
depends_on: []
---
# Goal

CUDA 機を使わず、この M5 Max の上で serving を詰める。packed（D2）と shared（D3）の engine を MLX で実装し、量子化した Qwen3-32B
（8 bit / 4 bit）で決定の出力が bf16 と実用上同じであることを確認したうえで、単一決定の p50 と S=8k / Q=100 のスループットを
PyTorch MPS の engine および同一機上の jevmlx と比較する。同じハードウェアでソフトウェアだけを変えたときの差を測り、
Benchmark Heaven の Speed 軸（p50 0.92 s、ローカル 0.65 s）に対してローカルで到達できる下限と、`/v1/systemone` の推奨構成を決める。

# Context

- PyTorch MPS の結果: S=8k / Q=100 で packed / shared は naive の 53〜73 倍。D3 は probe 用に SDPA が 2 回必要で packed 比 0.87 / 1.07 / 1.50 倍（1.7B / 14B / 32B）。
  32B の単一決定はローカル p50 0.65 s、Benchmark Heaven の測定ではドイツから 0.92 s（×2 補正で 1.85 s 相当）、Jev 本番 0.65 s。
- MLX: `uv pip install --dry-run` で mlx 0.32.2、mlx-lm 0.31.3、mlx-metal 0.32.2 が解決できる（未導入）。Hub に `mlx-community/Qwen3-32B-{4bit,6bit,8bit,bf16}` がある。
  MLX の SDPA はマスク配列を受け取れ、custom Metal kernel の API（`mx.fast.metal_kernel`）がある。バージョンで API 差があるので着手時に確認する。
- Qwen3-32B は head_dim 128、64 heads / 8 KV heads、64 層（MPS の SDPA は head_dim 128 が高速経路）。
- jevmlx（[bnsd55/jevmlx](https://github.com/bnsd55/jevmlx)、README 2026-09-21 確認）: MLX 上で context + schema を 1 回 prefill し、
  broadcast KV cache に対して option を trie 行として restricted softmax で採点する。`jevmlx serve` が `/decide` と `/v1/systemone` を出す。
  既定 alias は Qwen2.5 の 4 bit（7B / 3B / 1.5B）で、custom Hub ID も指定できる。TypeSafe の private set で 0.4 s/件（one Metal GPU）と記載。
  prompt と採点方式が jqv と違うので、比較するのは同一機での latency のみ（精度は比較しない）。
- 量子化の効き方: 短い state（S=500、Q=1）は帯域律速なので 4 / 8 bit が効く。長い prefill は計算律速なので効きは小さい。MPS の fp16 が bf16 より速い可能性も未計測。
- 量子化モデルは logits が変わるので、温度は val 400 で再学習して provenance を残す（`TemperatureScaler`）。

# Scope

## In

- `jqv/engine/mlx/`（naive / packed / shared）: mlx-lm の Qwen3 実装で bf16 と 4 / 8 bit を読み、packed は branch ごとの position と加算マスクで、
  shared は Hydragen 型の分解で実装する。まず PyTorch と同じ 2 パス（probe）構成、時間があれば custom Metal kernel で 1 パス化（1 日で打ち切り）。
  readout は `legal_mass` を出せるよう全語彙 softmax を保持する。
- 等価性: MLX bf16 と PyTorch bf16 の choice logits が bf16 許容内で一致、MLX 内で naive / packed / shared が一致（既存テストの MLX 版）。
  量子化は MMLU test 800 と JevBench public hard 111 の精度・ECE を bf16 と比較する（8 bit は ±0.5 pt を期待、4 / 6 bit は報告）。
- PyTorch MPS の fp16 vs bf16 の速度と精度（安価な確認）。
- ベンチ: S ∈ {500, 2000, 8000} × Q ∈ {1, 10, 100} で PyTorch packed / shared と MLX packed / shared × {bf16, 8 bit, 4 bit}。
  単一決定（S ≈ 1k、Q=1）の `/v1/systemone` end-to-end p50 / p95（ローカル、50 リクエスト）。
- jevmlx を同一機で起動し（`mlx-community/Qwen3-32B-4bit` が読めなければ既定の Qwen2.5-7B-4bit）、同じ 50 リクエストで latency を測る。
- サーバ: `JQV_BACKEND=mlx`（engine 選択と量子化の指定）。量子化モデル用の温度ファイル。
- README 節: 表、「同一ハードでソフトウェアだけでどこまで縮むか」の結論、推奨のローカル serving 構成。

## Out

- CUDA / H100、vLLM / SGLang。
- MLX 上での学習。
- prompt 形式の変更（`readout-codebook-legal-mass` で決まった形式をそのまま使う）。

# Success

- MLX 版 packed / shared の等価性テストが通り、8 bit の MMLU test 精度が bf16 の ±0.5 pt 以内（4 / 6 bit も表に載る）。
- 32B 単一決定の end-to-end p50 が精度を落とさない構成で 0.35 s 以下（現状 0.65 s。届かない場合はこの機で到達できない理由を記録）。
- S=8k / Q=100 の q/s で MLX の最良構成が PyTorch packed の 1.5 倍以上（届かない場合も表と理由を記録）。
- jevmlx の同一機 latency が同じ条件で表にあり、README に結論と推奨構成がある。
- ベンチと評価は進捗行と ETA を出し、途中結果を逐次保存する。

# Verify

```bash
uv add mlx mlx-lm
uv run pytest tests/test_mlx_equivalence.py                                              # 候補。実装後に確定
uv run python scripts/eval.py --dataset mmlu --backend mlx --model mlx-community/Qwen3-32B-8bit --engine packed   # --backend は候補
uv run python scripts/bench.py --backend mlx --engines packed,shared --quant 8bit --state-tokens 500 2000 8000 --questions 1 10 100
JQV_BACKEND=mlx JQV_MODEL=mlx-community/Qwen3-32B-8bit uv run uvicorn jqv.server:app --port 8000 &
PYTHONPATH=/Users/h.imura/tmp/repo/jevbench uv run python scripts/jevbench_run.py --tier easy ...      # 形式互換と精度の確認
uvx jevmlx serve --model mlx-community/Qwen3-32B-4bit                                     # 候補。モデル対応を確認
```

# Open Questions

- 既定で serve する量子化（8 bit が本命。4 bit は精度が保てれば）。
- custom Metal kernel で D3 を 1 パス化するか（時間枠 1 日。効果が薄ければ 2 パスのまま）。
