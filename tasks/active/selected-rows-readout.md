---
title: 語彙 projection なしの選択 row readout (B') が B と同一であることを示す
status: pending
priority: P2
created_at: 2026-09-20T01:50:40+09:00
depends_on: []
---

# Goal

語彙 15 万次元への projection を行わず、選択肢文字に対応する LM head の行だけから choice logits を計算する readout (B') を追加し、B（全語彙 projection → 抽出）と数値的に同一であることを示す。「vocab projection を捨てても決定は変わらない」という ablation の一段を埋める。

# Context

- `jqv/readout.py` の `decisions_from_hidden` は `lm_head(h)` で `(Q, V)` を計算してから `choice_ids` を抜いている。`choice_mass`（全語彙 softmax 上の選択肢質量）はこの全語彙 logits に依存する。
- B' は `h @ lm_head.weight[choice_ids].T` で同じ logits を得る。数学的に同一。
- 速度への寄与は小さい見込み（Q=100 で lm_head 約 31 GFLOP に対し backbone 約 46 TFLOP）。効果の有無は測って README に書く。

# Scope

## In

- `jqv/readout.py`: `readout="full" | "rows"` を選べるようにする。`rows` では `choice_mass` を `None` にする
- `jqv/engine/base.py`, `jqv/engine/__init__.py`: engine 生成時に readout モードを指定できるようにする
- `tests/test_engines_equivalence.py`: fp32 で `full` と `rows` の choice logits が 1e-4 以内で一致するテスト
- `scripts/bench.py`: `--readout` オプションを追加し、S=2038/Q=100 と S=538/Q=100 で packed の full/rows を比較
- README: engine 表に B' を追加し、等価性と速度差（有意差なしなら「なし」と書く）を記録

## Out

- 新規学習する decision head（`pointer-head-lora`）
- generate engine（生成に全語彙が必要）

# Success

- fp32 で全 engine（naive/kvcache/packed）の `rows` readout が `full` と同じ choice logits（差 < 1e-4）を返す
- `rows` モードで `choice_mass` が `None`、`full` モードで従来どおり値が入る
- README に B' の行と、full/rows の速度比較（2 条件以上）がある
- `uv run pytest` が全件成功する

# Verify

```bash
uv run pytest
uv run scripts/bench.py --engines packed --questions 100 --state-tokens 538 2038 --readout rows
uv run scripts/bench.py --engines packed --questions 100 --state-tokens 538 2038 --readout full
```

`--readout` は本タスクで追加するオプション。
