---
title: 現行 D3 の 2 回の SDPA を、attention 出力と logsumexp を同時に返す custom Metal kernel 1 回に置き換える（M5 Max 上の kernel lab）
status: draft
priority: P2
created_at: 2026-09-21T10:11:14+09:00
depends_on: []
---
# Goal

この M5 Max の上で、D3（shared-prefix attention）が shared prefix 側の分配関数を得るために払っている **2 回目の SDPA を消す**。
attention 出力 O と logsumexp L を同時に返す custom Metal streaming-attention kernel を独立した kernel lab として作り、
fp32 参照との数値一致、bf16 で現行 D3 と同じ decision logits、S=8k 領域での明確な高速化を示す。14B で end-to-end を確認し、
32B は最終ベンチだけ行う。既存の HF / PyTorch 実装は正しさの参照として残し、jqv 全体を MLX に移植はしない。

# Context

- 現行 D3（`jqv/engine/shared.py`）: Hydragen 型の分解。shared prefix について (O_s, L_s) = Attention(Q_b, K_s, V_s)、branch 内について (O_b, L_b) を計算し、
  L = log(e^{L_s} + e^{L_b})、O = e^{L_s − L} O_s + e^{L_b − L} O_b で合成する。MPS の SDPA は正規化後の出力しか返さないため、L_s を
  zero-key の probe を足した 2 回目の SDPA から逆算している（`_prefix_block_fused`、KEY_PAD=8）。これが D3 の主な損失で、packed 比 0.87 / 1.07 / 1.50 倍
  （1.7B / 14B / 32B）にとどまる。MPS の SDPA は head_dim ≠ 128 で約 10 倍遅い。
- 形状: Qwen3-14B は 40 heads / 8 KV heads / head_dim 128 / 40 層、Qwen3-32B は 64 / 8 / 128 / 64 層（GQA は KV head あたり query 5 または 8）。
- MLX の `mx.fast.metal_kernel(name, input_names, output_names, source, header, ensure_row_contiguous, atomic_outputs, compile_options)` は
  `output_names` で複数出力を返せ、`inputs, template, grid, threadgroup, output_shapes, output_dtypes` で呼ぶ（公式 doc を 2026-09-21 に確認）。
  mlx 0.32.2 / mlx-lm 0.31.3 が `uv` で解決できる（未導入）。
- PyTorch MPS と MLX はテンソルを直接共有しないので、microbenchmark は同じ入力を両方に渡して比較し（numpy 経由）、end-to-end は
  PyTorch から kernel を呼ぶのではなく mlx-lm の Qwen3 で最小の推論経路を組んで確認する。
- kernel の型: FlashAttention 型の streaming softmax（K/V を tile で読み、running max m・running sum l・running output o を更新し、最後に O と log l + m を出す）。
  参考実装: [manishklach/mlx-metal-kernels](https://github.com/manishklach/mlx-metal-kernels)（2026-08、MLX custom Metal の fast attention / decode / KV-cache の実験。
  生産用ではないが streaming attention の叩き台になる。着手時に内容を確認）。
- テストの雛形: `tests/test_engines_equivalence.py`（fp32 は一致、bf16 は許容差）。

# Scope

## In

- `jqv/metal/`: `reference.py`（fp32 の MLX 参照: SDPA + 別計算の logsumexp。計時用に現行 torch MPS D3 の attention 部分も関数として切り出す）、
  `shared_attention.py`（custom Metal kernel: tile 化した K/V の streaming softmax、GQA の head 対応、head_dim 128、bf16 入力・fp32 累積、出力 O と L、branch 内用の causal オプション）、
  `compose.py`（L と O の合成）。
- Phase 1 microbenchmark `scripts/bench_metal_attention.py`: 14B と 32B の形状、S ∈ {2k, 8k}、query rows ∈ {100, 1k}（余裕があれば 8k）で
  (1) 現行 MPS D3 の 2 回 SDPA、(2) MLX 参照、(3) custom kernel を比較。warmup 後 5 回の中央値、メモリも記録。
- テスト `tests/metal/test_shared_attention.py`: fp32 で kernel の (O, L) が参照と一致、bf16 で合成結果が torch D3 と既存の許容差内、GQA、causal、tile の倍数でない rows。
- Phase 2 end-to-end（14B）: mlx-lm の Qwen3-14B bf16 で state を prefill して層ごとの K_s / V_s を取り、branch token の forward で shared 側に kernel、
  branch 内に kernel（causal）または MLX SDPA を使い、末尾位置の choice token logits を取る。PyTorch D3 の decision logits と MMLU 50 問で比較（bf16 許容差）。
  S=2k / 8k × Q=10 / 100 / 1000 の速度。
- 32B: 最終ベンチのみ（同じ表）。
- `results/metal_attention.md` と README の D3 節の更新。

## Out

- jqv 全体の MLX 移植（学習、評価、サーバ）。backend と serving は `mlx-serving-quantization`。
- paged KV cache、decode 用 kernel、量子化 attention。
- prompt / readout の変更。

# Success

- fp32: 全テスト形状（GQA、causal 含む）で kernel の O と L が MLX 参照と最大絶対誤差 1e-4 以内（候補。着手時に確定）。
- bf16: 合成した shared attention から得た decision logits が現行 D3 と `test_engines_equivalence.py` の bf16 許容差内。14B end-to-end の 50 問で argmax が一致し logits が許容差内。
- (O, L) が層・ブロックごとに kernel 1 回で得られ、probe 用の SDPA がない。
- 速度: 14B 形状の S=8k / rows 1k で custom kernel が現行 MPS D3 attention の 1.5 倍以上（候補。最低条件は「速い」）。14B end-to-end の S=8k / Q=100 で PyTorch D3 より速い。
- `results/metal_attention.md` に 14B / 32B の表があり README が更新されている。
- 時間枠: 3 作業日で参照速度の 2 倍以内に入らない、または一致しない場合は負の結果として記録して止める。

# Verify

```bash
uv add mlx mlx-lm
uv run pytest tests/metal
uv run python scripts/bench_metal_attention.py --shape qwen3-14b --S 2048 8192 --rows 100 1000 --impl mps-d3,mlx-ref,metal
uv run python scripts/bench_metal_attention.py --shape qwen3-32b --S 8192 --rows 1000 --impl mps-d3,metal
uv run python scripts/bench_metal_e2e.py --model Qwen/Qwen3-14B --S 2048 8192 --Q 10 100 1000        # 候補。実装後に確定
```

# Open Questions

- tile サイズと threadgroup 構成（計測で決める）。
- branch 内 attention も kernel を使うか、MLX SDPA + 別計算の L で済ませるか。
