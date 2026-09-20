---
title: JevBench（public 231 決定）で jqv 1.7B / 14B / 32B を評価し Jev との差を MMLU と比較する
status: done
priority: P2
created_at: 2026-09-21T01:28:05+09:00
depends_on:
  - scaling-32b-final
---

# Goal

Jev 系 decision model 専用ベンチマーク JevBench の public 決定（easy 48 / standard 72 / hard 111 = 231 件）で jqv の
Qwen3-1.7B / 14B / 32B（B zero-shot、perm_avg、slot + LoRA）を評価し、tier 別精度・校正（top-label ECE と gold 分布との
total variation distance）・raw latency を README のスケーリング表に並べる。MMLU での Jev との差（0.918 vs 0.809 = 10.9 ポイント）が、
Decision 用途に近い JevBench hard（Jev 74.1%）で縮むのか広がるのかを確定する。

# Context

- ハーネス: https://github.com/fstandhartinger/jevbench （MIT。public 決定は `datasets/public/*.jsonl`、held-out 303 件は非公開で、
  Benchmark Heaven が bench request（GitHub issue、TypeSafe wire format の endpoint を指定）で測る）。
- リーダーボード: https://benchmarkheaven.com/jev-models （v1.2.5、2026-09-20 時点。Jev 1.13: Score 75.4 / Intelligence 90.4 /
  Calibration 82.7 / hard 74.1% / p50 0.65 s。SemIf Qwen3.5-4B: 74.7 / 85.9 / 72.6 / 59.5% / 0.20 s。GPT-5.6 Luna low: hard 94.5%）。
- スコア式: Score = (Intelligence × Calibration × Speed × Cost)^(1/4)。Intelligence = 重み付き精度（hard 30% / easy 14% / standard 28% / judge 28%）。
  Speed = 100 − 20·log10(s / 0.1 s)、self-hosted / demo endpoint は ×2 + 0.15 s の仮想補正（測定ではなく仮定）。
  Cost = 100 − 30·log10($ per 1,000 decisions / $0.001)。Calibration は hard tier の top-label 10-bin ECE と、gold が確率分布で与えられる
  問題（20 問）に対する予測分布との total variation distance の平均（ユーザー提供メモ。ハーネスの scoring コードで確認する）。
- 既存 adapter: `typesafe`（`/v1/systemone`、TypeSafe wire format）、`systemone_list`、`gradio_space`、`local_openjev`（in-process open weights）、
  `openai_compat`。新 adapter は base adapter を継承し task の serialize と response の parse を実装する。
- jqv 側: `POST /decision` は `{state, questions:[{question, choices}]}` → 確率。JevBench の question 型（choice / noul（yes-no）/ score・rubric）の
  うち、choice と noul は現状の K 択 readout で表現できる。score / rubric 型の扱いはハーネスの task 定義を読んで決める。
- 注意点（Benchmark Heaven 自身の但し書き）: 534 問の pilot、英語のみ、hard は Claude Opus 5 と GPT-5.6 Sol が作成し相互 review。
  「small models are very sensitive to option order」と明記されており、jqv の perm_avg の知見と一致する。
- 速度は raw 値で比較する（補正は仮定）。Cost は open model については hosting 想定料金であり、jqv では算出しない。

# Scope

## In

- ハーネスを `/Users/h.imura/tmp/repo/jevbench` に clone し、commit hash を記録。scoring コード（ECE / TVD の定義、tier 重み）を読んで README に正確に書く
- jqv に **TypeSafe wire format 互換 endpoint**（`/v1/systemone` 相当）を追加する（`jqv/server.py` または `jqv/systemone.py`）。
  choice / noul を jqv の Question に写像し、確率をそのまま返す。score / rubric 型は K 段階の choice に写像できるならそうし、できなければ未対応として記録。
  これによりハーネスの `typesafe` adapter がそのまま使え、将来 bench request（held-out 分）も出せる
- ハーネスをローカル（1 リクエストずつ）で実行: 3 backbone × {B zero-shot, perm_avg, slot+LoRA（1.7B: slot_lora、14B: qwen3-14b_slot_brier0、32B: qwen3-32b_slot_best）}。
  結果は `results/jevbench/<model>_<engine>/` に raw と results.jsonl、集計は `results/jevbench_summary.json`
- `scripts/jevbench_summary.py`: tier 別精度、Intelligence、Calibration（ECE、TVD）、raw latency p50 を表にし、Jev / SemIf のリーダーボード値（出典付き）を並べる
- README: 「JevBench」節（表、MMLU 差との比較、Benchmark Heaven の但し書き）。スケーリング表に JevBench hard 列を追加

## Out

- held-out 303 件（Benchmark Heaven への bench request は公開 endpoint が必要。Open Questions）
- Cost 軸の算出、他システムの実行、MLX

# Success

- `results/jevbench_summary.json` に 3 backbone × 3 engine の tier 別精度（easy / standard / hard、public 分）、Calibration（ECE、TVD）、raw p50 latency がある
- README の JevBench 節に上記の表と Jev（hard 74.1%）/ SemIf 4B（59.5%）の行が出典付きで並び、「MMLU の差 10.9 ポイントに対し JevBench hard の差は X ポイント」が書かれている
- jqv の TypeSafe 互換 endpoint に対してハーネスの `typesafe` adapter が改変なしで動く（`uv run pytest` に wire format のテストを追加）
- 32B（最良設定）の JevBench hard 精度が、Jev の 74.1% との差として明記されている

# Verify

```bash
git -C /Users/h.imura/tmp/repo/jevbench log -1 --format=%H
uv run python -m jqv.server --model Qwen/Qwen3-32B --engine packed --port 8000 &
cd /Users/h.imura/tmp/repo/jevbench && python -m jevbench.cli run --tasks datasets/public/original.jsonl --adapter typesafe --model jqv-qwen3-32b --base-url http://127.0.0.1:8000 --results RUN/results.jsonl --raw-dir RUN/raw
uv run scripts/jevbench_summary.py
uv run pytest
```

CLI の正確なオプション（`--base-url` の有無など）はハーネスの README / コードを読んで確定する。

# Open Questions

- held-out 分の評価のために公開 endpoint を立てて bench request を出すか（Cloudflare Tunnel 等。ユーザー判断）
- score / rubric 型の question を jqv でどう扱うか（K 段階 choice への写像で妥当か）

# Result

## Changed

- `jqv/systemone.py`, `jqv/server.py`: TypeSafe 互換 `POST /v1/systemone`（choice / noul / score、JSON state、複数質問の shared state）。`JQV_PERM_AVG` / `JQV_HEAD_DIR` でエンジン設定
- `scripts/jevbench_run.py`: サーバ起動 → ハーネス `typesafe` adapter で public 3 tier 実行 → summarize → 停止。`--force` は選択 tier だけ置換
- `scripts/jevbench_summary.py`: tier 別精度、Intelligence（public 正規化）、hard ECE（生 / MMLU val 温度転用）、生 p50、リーダーボード行
- `scripts/scaling_table.py`: JevBench hard 列
- `tests/test_systemone.py`（5 件）
- `results/jevbench/<label>/<tier>/`（9 label × 3 tier の results.jsonl / raw / summary.json / summary_T.json）、`results/jevbench_summary.{md,json}`
- `README.md`: 「JevBench での外部評価」節（表、family 別、結論 6 点、但し書き）
- ハーネスは `/Users/h.imura/tmp/repo/jevbench`（commit 7ce310c7262ed49cc85853339a8a42459298e3f3、リポジトリ外）

## Verified

- `uv run pytest`（systemone 5 件を含む）
- 9 設定 × 3 tier がすべて 111/111・72/72・48/48 で失敗 0（JSON state 修正前に止まった 4 label の hard tier は `--tiers hard --force` で再実行）
- `uv run scripts/jevbench_summary.py`、`uv run scripts/scaling_table.py`

## Deviations

- 温度は事後適用（推論時にサーバへ温度ファイルを渡していない）。argmax 不変のため精度は同一で、ECE のみ再計算。サーバに `JQV_TEMPERATURE_FILE` を渡す運用が本来の形
- 最初のチェーンは JSON state（dict）を 400 で拒否しており、1.7B × 3 と 14B packed の hard tier が 63/111 で停止したため修正後に再実行した
- score 型は level 番号 + 説明を選択肢として K 択に写像した（`0: none / 1: one / …`）。Success の「未対応なら記録」には該当せず全件応答

## Remaining

- held-out 303 問と judge tier（公開 endpoint を立てて Benchmark Heaven に bench request）
- long_policy / temporal_numeric の弱さへの対処（few-shot の policy 例題、perm_avg の hard 限定適用など）
