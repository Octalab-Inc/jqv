---
title: README の古い記述を最新結果に合わせ、冒頭に結論（Findings）節を置く
status: done
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

# Result

## Changed

- `README.md`: 冒頭に「結論（Findings）」4 点を追加。「5 つの推論構造 + 2 種類の readout」に修正。C 節（1.7B の decision training は校正改善、精度は JMMLU のみ有意、14B 以上で消失）、
  E 節（1.7B は λ=1 だったが 14B で λ=0 を採用、32B も λ=0）、Jev の値に付けていた「zero-shot」を「Hume の 1,200 問 MMLU サンプル、プロンプト条件不明」に変更、
  スケーリング節の「精度は backbone でほぼ決まる」→「4,800 例規模の decision training に比べ最大の要因は backbone scale」、校正の結論を「in-/near-distribution では
  scalar T で同水準、普遍的には転移しない」に変更、JevBench 節の温度転移を「部分転移（0.274 → 0.107、基準は未達）」に修正し 3 段階の distribution shift を明記、
  関連プロジェクトに bnsd55/jevmlx とアプリケーション区分（classifier.dev）を追加、「次フェーズ（未実装）」を「今後の課題」に改名
- `scripts/scaling_table.py`: Jev 行の注記から zero-shot を外す。表を再生成

## Verified

- Jev の値に「zero-shot」を結び付けた記述が残っていないこと（残る「zero-shot」は jqv 自身の条件を指す）
- 「結論（Findings）」節の存在、各節の数値が結果表と一致することを目視確認

## Deviations

- なし

## Remaining

- なし
