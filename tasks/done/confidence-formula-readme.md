---
title: confidence を Jev 互換の式にし README の主張を正確にする
status: done
priority: P1
created_at: 2026-09-20T01:50:39+09:00
depends_on: []
---

# Goal

API が返す `confidence` が Hume が TypeSafe 公式 adapter で確認した Jev 互換の式 `(p_max − 1/K) / (1 − 1/K)` になり、README が packed engine と Jev の関係、thinking 無効化、packed の計算量上の限界を正確に述べている状態にする。

# Context

- `jqv/readout.py` の `confidence_from_probs` は `1 − H(p)/log K`（entropy 版）を返している。Hume の式は max-probability 版。
- README の engine 表は packed を「Hume 推定の Jev 構造」と書いている。Hume 自身は「sibling isolation は tree mask 以外でも再現でき、exact な mask までは分からない」と留保している。
- `jqv/prompt.py` は assistant 側を `<think>\n\n</think>\n\n` で始める非 thinking プロンプトを固定し、`tests/test_prompt.py` が公式 `apply_chat_template(enable_thinking=False)` と一致することを検証済み。generate (A) も同じ prefix + suffix を使う。README にはこの事実が書かれていない。
- packed は `(1, 1, L, L)` の float mask を SDPA に渡すため、MPS では attention が dense に計算される。S=8038/Q=100 の理想比は線形層 60x・attention 36x で、実測 53x はその間。Jev 規模（state 23k × 5,000 問, L ≈ 173k）では mask だけで bf16 60GB になり成立しない。

# Scope

## In

- `jqv/readout.py`, `jqv/types.py`: `confidence` を max-probability 式に変更し、entropy 版を `entropy_concentration` として残す
- `tests/test_readout.py`: 一様分布で 0、one-hot で 1、K=2 の p=(0.75,0.25) で 0.5 になることを確認するテスト
- `README.md`: (1) engine 表の文言を「Hume の観測と整合する block/tree attention 実装の一つ」に修正 (2) `enable_thinking=False` 固定と A/B が同一プロンプトであることを明記 (3) packed を「計算グラフと KV 共有の正しさを検証する reference 実装」とし、上記の理想比・実測・Jev 規模の見積りを書く (4) `confidence` の定義を更新

## Out

- engine の計算の変更、bench の再実行
- D3（物理的に sparsity を使う実装）は `d3-shared-prefix-attention` で扱う

# Success

- `Decision.confidence` が一様分布で 0.0、one-hot で 1.0、K=2, p=(0.75, 0.25) で 0.5 になる
- `Decision.entropy_concentration` が従来の `1 − H(p)/log K` を返す
- README に上記 4 点が含まれ、「Hume 推定の Jev 構造」という表現が残っていない
- `uv run pytest` が全件成功する

# Verify

```bash
uv run pytest
grep -n "Hume 推定の Jev 構造" README.md   # 0 件であること
grep -n "enable_thinking" README.md          # 記載があること
```

# Result

## Changed

- `jqv/readout.py`: `confidence_from_probs` を `(p_max − 1/K)/(1 − 1/K)` に変更し、entropy 版を `entropy_concentration` として追加
- `jqv/types.py`, `jqv/engine/generate.py`: `Decision.entropy_concentration` を追加
- `tests/test_readout.py`: 式のテスト（一様→0、one-hot→1、K=2 p=(0.75,0.25)→0.5）
- `README.md`: engine 表の文言、`confidence` の定義、`enable_thinking=False` 固定の明記、packed が reference 実装であることと理想比・実測・Jev 規模の見積り

## Verified

- `uv run pytest`: 14 passed
- `grep "Hume 推定の Jev 構造" README.md`: 0 件、`grep enable_thinking README.md`: 記載あり

## Deviations

- なし

## Remaining

- なし
