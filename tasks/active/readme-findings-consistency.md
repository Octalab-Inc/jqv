---
title: README の古い記述を最新結果に合わせ、冒頭に結論（Findings）節を置く
status: pending
priority: P1
created_at: 2026-09-21T03:33:19+09:00
depends_on: []
---

# Goal

README が 32B と JevBench までの全結果と整合し、外部の読者が冒頭 2〜3 段落で研究の結論を把握できる状態にする。

# Context

ユーザーのレビュー（2026-09-21）で指摘された不整合: (1) 冒頭の「4 つの推論構造」は 5 構造 + 2 readout が正しい、(2) E 節の「14B / 32B は λ=1」は
14B sweep で λ=0 を採用した事実と矛盾、(3) C 節「decision training は 4,800 例でも効く」は MMLU の p=0.125 と 14B 以上での消失に対して強すぎる、
(4) Jev の 0.918 / 0.031 を「zero-shot」と書いているが Hume の記事から確認できるのは 1,200 問 MMLU サンプルと ECE 0.0313 まで、
(5) JevBench 節の「MMLU → JevBench hard への温度転移は成立した」は転移節で定めた基準（対角 + 0.02）を満たしていない（0.274 → 0.107、MMLU 内は 0.023）、
(6) 「精度は backbone でほぼ決まる」「32B + 温度 1 個で Jev の水準に届く」は限定を付けるべき、(7) 「次フェーズ（未実装）」に実装済みの C が残っている。
追加: 関連プロジェクトに bnsd55/jevmlx（MLX、確認済み）と、アプリケーション区分として classifier.dev（TypeSafe Jev を backbone にした分類サービス、確認済み）。

# Scope

## In

- README の上記 7 点の修正、冒頭「結論（Findings）」節（4 点）、関連プロジェクトの追加、「今後の課題」への改名
- `scripts/scaling_table.py` の Jev 行の注記（zero-shot 表記を外す）と表の再生成

## Out

- 新しい実験、コードの変更

# Success

- README に「zero-shot」を Jev の値に結び付けた記述が残っていない
- 冒頭に結論 4 点の節があり、各点に数値が付いている
- E 節・C 節・スケーリング節・JevBench 節の記述が結果表と矛盾しない

# Verify

```bash
grep -n "zero-shot" README.md | grep -i jev   # 0 件
grep -n "結論（Findings）" README.md
```
