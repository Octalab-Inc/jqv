---
title: CUDA / FlexAttention 版 D3 と prefix-caching serving で、Jev との速度差が serving 由来かを確かめる
status: draft
priority: P3
created_at: 2026-09-21T10:11:14+09:00
depends_on: []
---
# Goal

CUDA 環境で packed（block mask）と D3（shared-prefix attention）を FlexAttention / Hydragen 型のカーネルで動かし、vLLM または SGLang の
prefix caching と同じ `/v1/systemone` で比較して、Jev 本番 API との速度差のうち serving / ハードウェア由来の部分を切り分ける。
本番 endpoint を置く場合の設計（JevBench の row を順位付きにする条件でもある）を判断できる材料を出す。

# Context

- MPS の結果: S=8k / Q=100 で packed / shared は naive の 53〜73 倍。D3 は probe 用に SDPA が 2 回必要で、packed 比 0.87 / 1.07 / 1.50 倍（1.7B / 14B / 32B）。
- Benchmark Heaven の測定: ドイツからの p50 0.92 s、p95 4.71 s（×2 補正で 1.85 s 相当）に対し Jev 本番 0.65 s。
- 手元に CUDA 機はない。レンタル GPU（H100 / A100 1 枚）を想定し、時間と費用を Open Question にする。
- 参考実装: openjev-sglang（SGLang radix cache）、Hydragen、DeFT（README 関連プロジェクト節）。

# Scope

## In

- FlexAttention の BlockMask による packed engine（`jqv/engine/packed.py` の 4D mask をカーネル側の block-sparse mask に置き換え）。
- D3 の CUDA 版（probe なしで partition function を得る fused 実装、または Hydragen の分解を 1 パスで）。
- 等価性テスト（`tests/test_engines_equivalence.py`、fp32 一致 / bf16 許容）を CUDA で通す。
- `scripts/bench.py` を CUDA で実行（S ∈ {500, 2000, 8000} × Q ∈ {1, 10, 100}、14B / 32B）し、vLLM（prefix caching + choice token の logprobs）の同条件と並べる。
- 単一決定（S=2k、Q=1）の p50 と、`/v1/systemone` 経由の end-to-end p50 を測る。
- README 節（MPS との比較表、serving 由来の差の見積もり、本番 endpoint の設計案と GPU 時間 / 費用）。

## Out

- 常設の本番サービスの構築・運用。
- 学習。

# Success

- CUDA で等価性テストが通る。
- S=8k / Q=100 の q/s（packed-flex、D3-cuda、vLLM prefix caching）と、S=2k / Q=1 の p50 が表になっている。
- 「Jev との残りの速度差のうち serving 由来はどれだけか」の結論と、本番 endpoint に必要な構成・費用の見積もりが README にある。

# Verify

```bash
uv run pytest tests/test_engines_equivalence.py --device cuda
uv run python scripts/bench.py --device cuda --engines packed,shared,flex --state-tokens 500 2000 8000 --questions 1 10 100   # --engines flex は候補
uv run python scripts/bench_vllm.py ...   # 候補
```

# Open Questions

- GPU の調達先と予算（レンタル H100 / A100 を何時間か）。決まるまで着手しない。
- 本番 endpoint を置くか、リポジトリを公開するか（どちらかで JevBench の row が順位付きになる）。
