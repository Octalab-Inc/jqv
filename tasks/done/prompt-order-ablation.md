---
title: prompt 順序の ablation: shared prefill（state → Q）が question 条件付き表現（Q → state → Q）に対してどれだけ精度と校正を失っているかを JevBench で測る
status: done
priority: P1
created_at: 2026-09-24T13:14:12+09:00
depends_on: []
---

# Goal

jqv の設計判断「state を 1 回 prefill して多数の question を分岐させる（state → Q）」が、question を先に読ませて state を question 条件付きで
表現する layout（Q → state → Q）に比べてどれだけ精度と校正を失っているかを、同じ backbone・同じ readout で JevBench public tier（hard は family 別）と
MMLU で実測する。4 条件を比べ、A ≈ C なら shared prefill は損していない、C ≫ A なら query-conditioned representation を捨てている、B ≈ C > A なら
後段で question を繰り返すだけで足りる、と判定できる形にする。

# Context

- 現行の prompt（`jqv/prompt.py`）: prefix = system + "Document:\n{state}"、suffix = "Question / Options / Answer with the letter only" + 答えの cue。
  prefix は question に依存しないので packed / kvcache / shared engine が共有できる。prompt_hash は `PromptStyle` の全 field から作られ、温度ファイルと
  学習済み head はその hash に紐づく（不一致は拒否）。
- 4 条件（ユーザー指定）:

  | 条件 | prompt | shared state | 見たいもの |
  |---|---|---:|---|
  | A 現行 `state_first` | state → Q | ○ | baseline |
  | B `repeat_question` | state → Q → Q | ○ | 質問を 2 回入れる効果（shared prefill を保てる） |
  | C `query_first` | Q → state → Q | × | question 条件付き state の効果 |
  | D `query_first_only` | Q → state | × | 前置だけの効果 |

- JevBench の decision は 1 state に 1 question なので、C / D でも JevBench 上の計算量は A と同じ。共有が効かなくなるコストは多 question の bench
  （`scripts/bench.py`、S=2k・Q=100）で別に測る。
- 温度は layout ごとに MMLU val 400 で当て直す（`scripts/fit_temperature.py`、prompt_hash が違うので流用できない）。学習済み head（`32b-hardfam-v2`）は
  layout A で学習されており他 layout には載らないので、この ablation は zero-shot（packed / naive engine、vocabulary readout）で行う。
  勝った layout で head を学習し直すのは別タスク。
- 見るもの: 精度（tier 別、hard は family 別: 特に temporal_numeric / multi_hop / ambiguous / routing_hard）、同じ 111 問での対応比較（exact McNemar）、
  served T の ECE / Brier / ordinal MAE、MMLU 800 の精度と温度後 ECE。ECE が悪化して精度だけ上がるなら「logits が尖っただけ」と読む。
- 実行環境は GB10 2 台（`docs/gb10.md`）。32B zero-shot で layout 1 つあたり MMLU 1200 + JevBench 3 tier ≈ 25 分。

# Scope

## In

- `PromptStyle.layout`（4 値）と `PromptBuilder` の対応（B は suffix に question block を 2 回、C / D は prefix が question に依存する）。
  `DecisionEngine.decide` で prefix が question 依存の layout のときは question ごとに prefix を作る（全 engine で動く）。
  `load_runtime` / `--layout` / `JQV_LAYOUT`（server）/ `jevbench_run.py --layout` / `eval.py` の出力 tag。テスト（prompt 文字列の順序、hash の差、engine 等価性）。
- 32B zero-shot で 4 layout × (MMLU 1200 → 温度 → JevBench easy / standard / hard)。family 別表、対応比較、校正指標。
- 多 question の速度: `scripts/bench.py` で A（packed、共有）と C（question ごと）を S=2038・Q=100 で比較（14B または 32B）。
- `docs/report.ja.md` / `docs/report.md` に節を追加（結論の 3 分岐のどれに当たるか）。

## Out

- 学習済み head の layout 別再学習（結果を見て別タスク）。
- prompt の文言変更（順序以外）。
- JevBench への提出。

# Success

- 4 layout の JevBench public 3 tier の精度、hard の family 別、A との対応比較（勝敗数と exact McNemar p）、served ECE / Brier / ordinal MAE、
  MMLU 精度と温度後 ECE が 1 つの表にある。
- 多 question 時の速度差（A vs C）が測られている。
- 結論が A ≈ C / C ≫ A / B ≈ C > A のどれに当たるか、family 別の根拠付きで report に書かれている。
- `uv run pytest tests/test_prompt.py tests/test_calibration.py` が通り、既定 layout の prompt_hash（4f85a0b34776）が変わらない。

# Verify

```bash
uv run pytest tests/test_prompt.py tests/test_calibration.py tests/test_engines_equivalence.py -q
uv run python scripts/eval.py --dataset mmlu --n 1200 --n-val 400 --model Qwen/Qwen3-32B --engine packed --layout query_first
uv run python scripts/fit_temperature.py results/mmlu_packed_qwen3-32b_layout-query_first.npz
JEVBENCH_DIR=~/jevbench PYTHONPATH=~/jevbench uv run python scripts/jevbench_run.py --model Qwen/Qwen3-32B --engine packed --layout query_first \
  --temperature-file results/mmlu_packed_qwen3-32b_layout-query_first_temperature.json --label qwen3-32b_packed_T_query_first
uv run python scripts/jevbench_families.py --labels qwen3-32b_packed_T_gb10 qwen3-32b_packed_T_repeat_question qwen3-32b_packed_T_query_first qwen3-32b_packed_T_query_first_only
```

# Open Questions

- なし（条件と指標はユーザー指定）。

# Result

- Changed: `jqv/prompt.py`（`PromptStyle.layout`、`LAYOUTS`、`question_block` / `_tail`、per-question prefix、`suffix_ids_with_spans` は最後の block を見る。
  既定 layout の prompt_hash 4f85a0b34776 は不変）、`jqv/engine/base.py`（query-first layout では question ごとに prefix を作る）、`jqv/engine/head.py`、
  `jqv/model.py`（`load_runtime(layout=)`、`default_style_for(layout=)`）、`jqv/server.py`（`JQV_LAYOUT`、/health に layout）、`jqv/systemone.py`（入力 token 数を
  question ごとに数える）、`scripts/_common.py`（`--layout`）、`scripts/eval.py`（出力 tag `_layout-<L>`）、`scripts/jevbench_run.py`（`--layout`）、`scripts/bench.py`、
  `scripts/prompt_order_table.py`（新規）、`tests/test_prompt.py`（4 本）、`tests/test_systemone.py`（1 本）。結果: `results/jevbench/qwen3-32b_packed_T_ablate-<L>/`（4 layout × 3 tier）、
  `results/mmlu_packed_qwen3-32b_{ablate-a,layout-*}*`、`results/compare_mmlu_32b_layouts.json`、`results/prompt_order_ablation_qwen3-32b.{md,json}`、
  `results/bench_qwen3-32b_layout-{state_first,query_first}.jsonl`。`docs/report.ja.md` / `docs/report.md` の節、README Findings 9。
- Verified: 4 layout の JevBench 3 tier、hard の family 別、A との exact McNemar、served ECE / Brier / ordinal MAE、MMLU の精度・T・ECE が 1 つの表にある
  （`results/prompt_order_ablation_qwen3-32b.md`）。多 question の速度差（A vs C、Q=1/10/100）を測った。`uv run pytest tests/test_prompt.py tests/test_calibration.py
  tests/test_engines_equivalence.py tests/test_isolation.py tests/test_systemone.py` が通り、既定 hash は不変。結論は **A ≈ C**（hard 68 → 71 / 111、7 勝 4 敗、p=0.55；
  MMLU では B ≈ C > A、+1.9 pt、p≈0.03；D は hard −4）。
- Deviations: server 経路（`decide_systemone`）と `bench.py` が query-first layout で prefix を question なしで作ろうとして落ちたため、修正して C / D の JevBench と
  bench を再実行した（eval 経路と MMLU は最初から正常）。A の JevBench は host A で再測定（以前の host B の zero-shot と 111 問すべて一致）。
- Remaining: 学習済み head を C で学習し直して A の head と比べる、A の shared prefill を保ったまま分岐内に question 条件付きの pass を足す折衷案。
  いずれも期待値は hard +3 / 111 程度で、多 question の 7.5〜18 倍のコストに見合わない限り優先度は低い。
