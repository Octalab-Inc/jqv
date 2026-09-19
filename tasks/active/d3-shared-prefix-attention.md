---
title: L×L mask なしで shared prefix attention を物理的に共有する engine (D3) を追加する
status: pending
priority: P2
created_at: 2026-09-20T01:50:45+09:00
depends_on: []
---

# Goal

L×L の明示 mask を持たず、shared prefix への attention を全 branch でまとめて計算し、branch 内の attention だけを個別に計算する engine (D3) を追加する。packed (D2) と数値的に一致しつつ、attention の演算量とメモリが block sparsity に比例して減ることを示し、D2 では扱えない規模（S=8k × Q=1000）を完走させる。

# Context

- `jqv/engine/packed.py` は `(1, 1, L, L)` の float mask を SDPA に渡す。MPS では dense に計算され、Jev 規模（state 23k × 5,000 問, L ≈ 173k）では mask だけで bf16 60GB になる。
- Transformers 5.17 では `AttentionInterface.register(name, fn)` でカスタム attention 関数を登録でき（`.venv/lib/python3.12/site-packages/transformers/modeling_utils.py` の `class AttentionInterface`）、`model.set_attn_implementation(name)` で切り替えられる。登録名が mask 生成の対応表にない場合、モデルは attention_mask を作らず `None` を渡す。branch の layout は attention module の属性として渡す。
- Hydragen の分解: branch の query をまとめて prefix KV に 1 回 attention（mask 不要、(Σq) × S）、branch 内は causal attention（Σ q_i²）、両者を log-sum-exp で合成。prefix 行は通常の causal attention。
- Qwen3-1.7B は GQA（q 16 heads, kv 8 heads）。`transformers.integrations.sdpa_attention.repeat_kv` と同じ扱いが必要。
- FlexAttention（`torch.nn.attention.flex_attention`）は import できるが CUDA 前提で MPS では実行できない。

# Scope

## In

- `jqv/engine/d3.py`（仮称）: custom attention を登録し、packed と同じ入力（prefix + branch 群、branch ごとの position id）で forward する engine。prefix cache + chunk にも対応
- MPS backend: 上記 Hydragen 型分解を純 PyTorch（matmul + softmax、または sdpa + 手計算 LSE）で実装。branch を chunk に分けてメモリを抑える
- CUDA backend: 同じ engine 内で FlexAttention の `BlockMask` を使う実装を用意し、CUDA 未検証と README に明記する
- `tests/test_engines_equivalence.py`: fp32 で naive と 1e-3 以内
- `tests/test_isolation.py`: D3 でも兄弟質問の秘密が漏れない
- `scripts/bench.py`: D3 を engine 一覧に追加。S=8038 で Q=100 と Q=1000 を D2（`max_tokens` 超の chunk 動作）と比較し、peak memory（`torch.mps.current_allocated_memory` 等）も記録
- README: D3 節（分解の式、D2 との差、測定結果）。engine 表を A/B/B'/D1/D2/D3 に更新

## Out

- vLLM / paged attention との統合
- 学習

# Success

- D3 の choice logits が naive と fp32 で 1e-3 以内、isolation テストが通る
- S=8038 / Q=1000 で D3 が完走し、D2 との時間と peak memory の比較が README にある
- D3 が保持する attention 行列の最大サイズが (Σq) × S と max(q_i)² のオーダーであることがコード上で明らか（L×L を作らない）
- MPS で fused SDPA より遅い場合も、その結果と理由を README に記録する

# Verify

```bash
uv run pytest
uv run scripts/bench.py --engines naive kvcache packed d3 --state-tokens 8000 --questions 100 1000
```

`d3` の engine 名は本タスクで確定する。

# Open Questions

- MPS では手書きの matmul + softmax が fused SDPA より遅い可能性がある。結果は結果として記録し、CUDA 版での再測定を次の判断材料にする
