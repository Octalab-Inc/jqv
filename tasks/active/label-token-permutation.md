---
title: 文字ラベル (A/B/C/D) token の prior を位置固定の入れ替えで定量化する
status: pending
priority: P2
created_at: 2026-09-20T01:50:43+09:00
depends_on: []
---

# Goal

選択肢の位置と内容を固定したまま、前に付ける文字ラベル（A/B/C/D）だけを入れ替えたとき、意味的選択肢の確率がどれだけ動くかを測り、文字 token 自体の prior を定量化する。

# Context

- `jqv/prompt.py` の `suffix_text` は位置順に `A.`, `B.`, `C.` … を付け、readout は文字 → 位置で対応させている。ラベルの順序は変えられない。
- Hume は Jev で option order によって確率が動くことを観測している。語彙 readout では「文字 token の prior」と「listwise の順序効果」が混ざるので、本タスクで前者だけを分離する。後者は `option-order-permutation` で扱う。
- 評価データは `jqv/data.py` の `load_named("mmlu")`（4 択）と `bridge`。

# Scope

## In

- `jqv/prompt.py`: `labels: list[str] | None` を受け取り、位置 i に `labels[i]` を付けられるようにする（既定は A, B, C …）。readout の choice_ids も同じ順で解決する
- `scripts/permutation_test.py --mode label`: MMLU 300 問（seed 固定）× ラベルの巡回シフト 4 通り、bridge 全件 × K 通り。各 permutation の確率を意味的選択肢に戻して集計
- 指標: (1) 文字ごとの平均確率（内容を平均した letter prior） (2) 同一問題での p(correct) の permutation 間の平均絶対差 (3) argmax が全 permutation で一致する問題の割合 (4) permutation ごとの accuracy
- `results/permutation_label_qwen3-1.7b.json` と README 節
- `tests/test_prompt.py`: labels 指定時のプロンプトと choice_ids の対応テスト

## Out

- 選択肢テキストの並び替え（`option-order-permutation`）
- 5 番目選択肢実験の再測定（同上）

# Success

- README に letter prior 表（A..D への平均確率）、p(correct) の平均絶対差、argmax 一致率、accuracy の範囲がある
- 「文字 token の prior は無視できる／できない」の結論が数値付きで書かれている
- `uv run pytest` が全件成功する

# Verify

```bash
uv run pytest
uv run scripts/permutation_test.py --mode label --n 300
cat results/permutation_label_qwen3-1.7b.json
```
