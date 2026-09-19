---
title: 選択肢の並び順による確率変動と 5 番目選択肢効果を n=300 で定量化する
status: done
priority: P2
created_at: 2026-09-20T01:50:44+09:00
depends_on:
  - label-token-permutation
---

# Goal

文字ラベルを A, B, C, D の順に固定し、選択肢テキストの並びを巡回シフトしたときの意味的確率の変動（listwise の順序効果）を測り、`label-token-permutation` の結果と対比する。あわせて 5 番目の無関係選択肢を足す実験を n=300 で取り直す。

# Context

- 現状の 5 番目選択肢実験（`scripts/isolation_test.py`）は bridge 21 問で −0.41 ± 0.41 と信頼区間が広すぎる。
- `label-token-permutation` で作る `scripts/permutation_test.py` と集計指標を再利用する。
- Hume の Jev では option order で確率が動き、5 番目選択肢で既存選択肢間の log-odds が −0.28 (95% CI −0.36〜−0.19) 動いた。

# Scope

## In

- `scripts/permutation_test.py --mode order`: MMLU 300 問 × 選択肢テキストの巡回シフト 4 通り（ラベルは固定）。指標は label モードと同じ（position prior、p(correct) の平均絶対差、argmax 一致率、accuracy の範囲）
- `scripts/permutation_test.py --mode fifth`: MMLU 300 問で「該当なし／不明」を 5 番目に追加し、正解 vs 最有力誤答の log-odds 変化の平均と 95% CI
- `results/permutation_order_qwen3-1.7b.json`, `results/fifth_option_qwen3-1.7b.json` と README 節（label と order を並べた表）

## Out

- プロンプト形式の変更（順序効果を減らす工夫は本タスクの対象外）

# Success

- README に position prior 表と、label / order 両モードの平均絶対差・argmax 一致率を並べた表がある
- 5 番目選択肢の Δlog-odds が n=300 の 95% CI 付きで README にあり、Hume の −0.28 との関係が書かれている
- 「順序効果は文字 prior より大きい／小さい」の結論が数値付きで書かれている

# Verify

```bash
uv run scripts/permutation_test.py --mode order --n 300
uv run scripts/permutation_test.py --mode fifth --n 300
```

# Result

## Changed

- `scripts/permutation_test.py`: `--mode order`（並びだけ巡回）と `--mode fifth`（5 番目追加、Δlog-odds と 95% CI、5 番目の確率質量、argmax flip 率）を計測
- `README.md`: 「選択肢の並び順と 5 番目選択肢」節（label と order を並べた表、fifth の表）。旧 n=21 の記述を差し替え
- `results/permutation_order_qwen3-1.7b.json`, `results/permutation_fifth_qwen3-1.7b.json`

## Verified

- `uv run scripts/permutation_test.py --mode order --n 300`、`--mode fifth --n 300` が完走し JSON を生成

## Deviations

- fifth の 5 番目文言「該当なし／不明」は日本語のままで英語 MMLU にも使った（isolation_test.py と同じ文言で比較可能にするため）。5 番目に 19% の確率が乗る一因になっている可能性を README に注記

## Remaining

- 選択肢文言の言語を揃えた fifth の再測定
