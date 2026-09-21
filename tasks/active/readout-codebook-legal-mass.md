---
title: jevmlx 型の readout 改善（neutral codebook、legal_mass、none-of-above、abstention）を /decision に入れる
status: draft
priority: P2
created_at: 2026-09-21T10:11:14+09:00
depends_on: []
---
# Goal

`/decision` の readout を jevmlx の実用機能に合わせて拡張する: 文字 A/B/C の代わりに tokenizer とラベル prior を見て選ぶ neutral codebook、
許可された継続トークンへの確率質量 `legal_mass`、`probability_margin`、明示的な none-of-above、閾値による abstention。
文字ラベルの prior による偏りを減らし、「候補の中では自信があるが候補から答えたがっていない」ケースを検出できるようにする。

# Context

- 現在の readout は `jqv/prompt.py` の `Answer:` 直後の単一トークン `" A"`, `" B"`, … の logits（`jqv/readout.py`）。文字ラベルの prior は
  `scripts/permutation_test.py --mode label` で測定済み（README「文字ラベルの prior」節）。
- jevmlx（[bnsd55/jevmlx](https://github.com/bnsd55/jevmlx)、README 2026-09-21 確認）: 「codebook search picks the neutral alias codes whose candidate rows
  tokenize most cleanly」、`legal_mass`（probability mass in allowed continuations — a leakage signal when low）、`allow_none_of_above=True`
  （NONE_OF_ABOVE を追加して None に写像）、`abstain_below_margin=X`（`probability_margin` = top1 − top2 が閾値未満なら保留）、
  `calibrate` CLI（temperature + multi calibrator）。
- `Question.labels`（`jqv/types.py`）で任意ラベルを与えられる。codebook を変えると `prompt_hash` が変わるので温度は再学習が要る
  （`TemperatureScaler.check_compatible` が不一致を検出する）。
- `/v1/systemone` の wire format（TypeSafe 互換）は変えない。追加フィールドをハーネスが許容するかは easy tier の実行で確認する。
- 後続の `correctness-calibrator` は `legal_mass` と `probability_margin` を特徴量に使うので、`scripts/eval.py` の npz にこれらを保存する。

# Scope

## In

- codebook: 候補集合（例: 数字、短い記号、jevmlx 型の alias code）から、単一トークンに tokenize され、校正用 dev（MMLU val 400）での
  ラベル周辺分布の偏り（permutation 平均での max − min）が最小になる集合を選ぶ `jqv/codebook.py`。`PromptBuilder` の `labels` として使う。
- `legal_mass`: readout 位置の全語彙 softmax のうち K 個の許可トークンに載る質量。`Decision` に `legal_mass` と `probability_margin` を追加する。
- `/decision` のオプション `allow_none_of_above`（escape ラベルを追加し、選ばれたら `decision: null`）と `abstain_below_margin` /
  `abstain_below_legal_mass`（保留時は `abstained: true` と理由）。サーバの既定値は env で指定。
- `scripts/eval.py --save-features`（p_max、entropy、margin、legal_mass を npz に保存）。
- 測定（1.7B と 32B、MMLU test 800 と JevBench public hard 111）: 文字 vs codebook の精度と prior 偏り、`legal_mass` の分布
  （MMLU / bridge / JevBench hard）、`legal_mass` と正誤の関係（AUROC）、abstention を入れた選択的精度（同じ coverage で precision 比較）。
- テスト（codebook が単一トークンであること、legal_mass の範囲、abstention の分岐、systemone 形式が不変であること）と README 節。

## Out

- `/v1/systemone` の出力形式の変更（追加フィールドの許容確認のみ）。
- 学習（`hard-family-targeted-lora`）、calibrator の学習（`correctness-calibrator`）。
- 複数選択（multi-select）や順序付き enum の telemetry。

# Success

- codebook で 1.7B と 32B のラベル prior 偏り（permutation 平均の max − min）が文字ラベル比で半分以下になり、MMLU test 精度が ±0.5 pt 以内か改善。
- `legal_mass` と `probability_margin` が `/decision` と eval の npz に出る。`legal_mass` が低い決定は誤りが多い（AUROC > 0.6）ことを 32B で確認、または反証を記録。
- abstention: p_max 閾値と同じ coverage で precision が同等以上。
- 既存 41 テスト + 新規テストが通り、`/v1/systemone` 経由のハーネス easy tier が 100% のまま。
- 32B の codebook 用の温度を再学習して provenance 付きで保存してある。

# Verify

```bash
uv run pytest
uv run python scripts/permutation_test.py --mode label --model Qwen/Qwen3-1.7B --codebook neutral      # --codebook は候補
uv run python scripts/eval.py --dataset mmlu --model Qwen/Qwen3-32B --engine packed --codebook neutral --save-features
uv run python scripts/fit_temperature.py --cache results/mmlu_packed_qwen3-32b_neutral.npz
curl -s -X POST localhost:8000/decision -d '{"state":"...","questions":[{"question":"...","choices":["a","b"]}],"allow_none_of_above":true}'
PYTHONPATH=/Users/h.imura/tmp/repo/jevbench uv run python scripts/jevbench_run.py --tier easy ...             # 形式互換の確認
```

# Open Questions

- codebook の候補集合をどこまで広げるか（数字 / 記号 / 単語）。測定で決める。
- abstention の既定閾値をサーバ既定にするか、クライアント指定のみにするか。
