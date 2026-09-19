---
title: 学習なしで直接 readout の精度を上げる（few-shot state と選択肢巡回シフト平均）を 1.7B で測り engine に組み込む
status: done
priority: P2
created_at: 2026-09-20T05:23:54+09:00
depends_on: []
---

# Goal

モデルを学習せずに直接 readout（B / packed）の精度を上げる 2 つの手段を、同じ 1,200 問サブセットで定量化し、効くものを engine のオプションとして
組み込む。目標は MMLU で 80% 前後だが、本タスクは 1.7B で「何ポイント効くか」を確定し、14B / 32B の評価にそのまま使えるようにするところまで。

# Context

- 現状（Qwen3-1.7B、zero-shot、非 thinking）: MMLU 0.554、JMMLU 0.466。順序を巡回シフトすると 52% の問題で argmax が変わる
  （`README.md` 「選択肢の並び順」）。つまり決定の半分は位置・文字 prior の tie-break で決まっており、ここを平均化すれば精度が上がる余地がある。
- (a) few-shot: MMLU の公式評価は 5-shot。jqv では例題を **shared state（prefix）** に置けるので、packed / shared engine なら prefill は
  1 回で済み、質問ごとのコストは増えない。例題は MMLU `dev`（285 問、subject ごと 5 問）から同 subject の 5 問を使うのが標準。
  subject をまたぐ固定 5 問（1 つの state を全質問で共有する Jev 型）も測る。
- (b) 巡回シフト平均: 選択肢の並びを K 通り（K=選択肢数）巡回させ、意味的選択肢に戻して確率を平均する。コストは K 倍だが state は共有される。
  `scripts/permutation_test.py --mode order` の集計コードを再利用できる。
- 評価は `scripts/eval.py`（seed 0、n 1200、n-val 400）と `scripts/compare_runs.py`（対応比較）で行う。

# Scope

## In

- `jqv/engine/base.py`（または `DecisionEngine.decide` のラッパ）: `perm_avg=True` で K 通りの巡回シフトを 1 回の `decide` にまとめて
  推論し、確率と logits（log の平均）を意味的順序に戻して返す。`Decision.probabilities` の和は 1 のまま
- `jqv/data.py` / `scripts/eval.py`: `--shots N`（同 subject の dev から N 問を state に入れる）と `--shots-fixed N`（全質問共通の N 問）。
  state 文字列の書式（例題の並べ方）は `jqv/prompt.py` に関数として置く
- 1.7B で 4 条件（zero-shot / 5-shot 同 subject / 5-shot 固定 / perm_avg）+ 組合せ（5-shot 同 subject + perm_avg）を MMLU と JMMLU で評価し、
  `compare_runs.py` で B（zero-shot）との対応比較（McNemar）を出す。温度 fit と ECE も記録
- `scripts/bench.py` に `--perm-avg` を追加し、コスト（q/s）の変化を 1 条件（S=2038, Q=100, packed）で測る
- README: 「学習なしの精度向上」節（表、対応比較、コスト）

## Out

- 学習（LoRA）や head の変更
- thinking / CoT 生成（非生成の原則から外れる）
- 14B / 32B での実行（`scaling-14b-baseline` 以降で本タスクの最良設定を使う）

# Success

- MMLU / JMMLU（各 test 800）で 5 条件の accuracy / ECE（raw, +T）と、zero-shot に対する Δ と McNemar p が README の表にある
- `perm_avg` と `--shots` が eval.py / server 経由で使えて、`uv run pytest` が全件成功する（perm_avg で確率の和が 1、K=2 の対称性テスト）
- 「14B / 32B に持っていく設定」（few-shot の有無・方式、perm_avg の有無）が根拠付きで README に書かれている

# Verify

```bash
uv run pytest
uv run scripts/eval.py --dataset mmlu --engine packed --n 1200 --n-val 400 --shots 5
uv run scripts/eval.py --dataset mmlu --engine packed --n 1200 --n-val 400 --perm-avg
uv run scripts/eval.py --dataset mmlu --engine packed --n 1200 --n-val 400 --shots 5 --perm-avg
uv run scripts/compare_runs.py --dataset mmlu --runs "zero-shot=packed_qwen3-1.7b" "5-shot=packed_qwen3-1.7b_shots5" ...
```

結果ファイルのタグ（`_shots5`, `_permavg`）は本タスクで確定する。

# Result

## Changed

- `jqv/fewshot.py`: MMLU dev からの同 subject 例題 / 固定例題で shared state を作る
- `jqv/engine/base.py`: `perm_avg`（K 通りの巡回シフトを 1 回の decide にまとめ、意味的順序に戻して確率を平均。`logits` は log(mean p)）。全 engine と HeadEngine / GenerateEngine が `_decide_plain` 経由で対応
- `jqv/types.py`: `Decision.perm_avg_k`
- `scripts/eval.py`: `--shots`, `--shots-mode subject|fixed`, `--perm-avg`、結果行を元の item 順で保存（state ごとの grouping で順序がずれ対応比較が壊れていたバグを修正）
- `scripts/bench.py`: `--perm-avg`（`packed:permavg` 行）
- `tests/test_permavg.py`: 確率の和、巡回不変性、few-shot 書式
- `README.md`: 「学習なしの精度向上」節（表、対応比較、コスト、固定例題の文字 prior 誘導）

## Verified

- `uv run pytest`: 35 passed
- MMLU / JMMLU 各 5 条件（test 800）+ bridge perm_avg、`fit_temperature.py`、`compare_runs.py`（McNemar）、bench 1 条件

## Deviations

- eval.py の行順バグにより few-shot 系 6 run を再実行した（bf16 のため再実行前後で 1 ポイント程度の差）
- JMMLU の few-shot 例題は英語（MMLU dev）のまま

## Remaining

- 巡回シフト以外の順序集約（全順列、部分集合）は未検証
