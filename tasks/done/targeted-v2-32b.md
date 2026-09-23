---
title: 32B を v2 mix で学習し、gate を通れば jqv-targeted として JevBench に 1 回だけ再提出する
status: done
priority: P1
created_at: 2026-09-23T09:56:52+09:00
depends_on:
  - temporal-generator-v2
---

# Goal

14B で独立 gate を通した学習設計（temporal generator v2、mix long_policy 0.25 / temporal v1 0.10 / temporal v2 0.15 / probability 0.20 / MMLU 0.30）を
32B にそのままスケールし、JevBench public hard で v1 の 32B head（72 / 111、temporal_numeric 2 / 15）を上回るか、temporal_numeric が回復するかを見る。
gate を通った場合のみ、Benchmark Heaven に `jqv-targeted` として別 row で 1 回だけ再提出する（zero-shot の既存 row は残す）。gate に届かなければ提出しない。

# Context

- 14B の結果（`temporal-generator-v2`）: JevBench hard 61 → 67（v1）→ 74 / 111（v2、zero-shot 比 22 勝 9 敗、p=0.029）、temporal_numeric 5 → 3 → 6 / 15、
  long_policy 5 → 8 → 12、judge_hard 14 → 12。synth temporal_v2 test 0.244 → 0.614、MMLU 0.764。
- 32B の対比: zero-shot 68 / 111（GB10）、v1 head 72 / 111（temporal 2 / 15、probability 8 / 10、ECE 0.096、Brier 0.415）、MMLU 0.821、JMMLU 0.779。
- recipe（v1 32B と同一）: slot + LoRA r=16、λ=0、600 step × batch 8（micro 2 × 累積 4、gradient checkpointing）、max_len 4096。GB10 A で 50 秒/step、約 8.5 時間。
  データは GB10 A の `data/synth/temporal_v2/train.jsonl`（md5 2494b8f1954b、seed 0 + `train.paraphrase.jsonl` で再現）。
- 評価: B で synth 4 family test（slot、32B）と v1 32B head の temporal_v2 test 対照、A で MMLU / JMMLU 1200 → 温度 → JevBench hard（`--label qwen3-32b_hardfam_v2_T`）。
  比較は Mac で `compare_runs.py`、`jevbench_families.py`、exact McNemar。runbook は `docs/gb10.md`。
- 提出: 前回は公開コードから Benchmark Heaven 側が再実行した（jevbench#9）。targeted head は checkpoint（adapter + head、約 150 MB）の公開先が要る（GitHub release か HF Hub。ユーザーに確認）。

# Scope

## In

- 32B の学習（A、run `32b-hardfam-v2`）と評価（B: synth 4 family + 対照、A: MMLU / JMMLU / JevBench hard）、結果の回収と比較、`docs/report.ja.md` / `docs/report.md` への記録。
- gate 判定（下記 Success）。通過時: `docs/jevbench-serving.md` に targeted 構成（`--engine slot --head-dir` + 温度）を追記し、checkpoint を公開して jevbench issue で `jqv-targeted` の row を依頼する。

## Out

- recipe や mix の変更、生成器の追加変更、RLCD。
- gate 未達時の提出。
- zero-shot row の差し替え。

# Success

- JevBench public hard が v1 の 72 / 111 を上回る。
- temporal_numeric が v1 の 2 / 15 から回復し、最低でも zero-shot の 4 / 15 以上（目標 6 以上）。
- probability が 8 / 10 前後を維持する。
- long_policy を壊さず、MMLU / JMMLU に明確な退行がない（−1 pt 以内）。
- served ECE / Brier が v1 の 0.096 / 0.415 より悪化しない。
- 上記を満たしたときだけ `jqv-targeted` として 1 回提出する。

# Verify

```bash
ssh gb10a 'tail -3 ~/train32v2.log; grep val_acc ~/train32v2_run.log | tail -6'
uv run python scripts/jevbench_families.py --labels qwen3-32b_packed_T_gb10 qwen3-32b_hardfam_T qwen3-32b_hardfam_v2_T
uv run python scripts/compare_runs.py --dataset synth-temporal_v2-test --runs "zero-shot=packed_qwen3-32b" "v1 head=slot_qwen3-32b_32b-hardfam-gb10" "v2 head=slot_qwen3-32b_32b-hardfam-v2"
uv run python scripts/compare_runs.py --dataset mmlu --runs "zero-shot GB10=packed_qwen3-32b_gb10" "v1 head=slot_qwen3-32b_32b-hardfam-gb10" "v2 head=slot_qwen3-32b_32b-hardfam-v2"
uv run python scripts/compare_runs.py --dataset jmmlu --runs "zero-shot GB10=packed_qwen3-32b_gb10" "v1 head=slot_qwen3-32b_32b-hardfam-gb10" "v2 head=slot_qwen3-32b_32b-hardfam-v2"
```

# Decisions（2026-09-23、ユーザー）

- 条件は提案どおり（mix、A で学習、B で synth、A で MMLU / JMMLU / JevBench）。
- 今は提出せず、32B v2 の結果を見てから 1 回だけ再提出。gate は Success のとおり。72 前後に留まるなら見送る。

# Result

- Changed: `docs/jevbench-serving.md`（「Targeted configuration」節: checkpoint の取得、環境変数、/health の期待値、public tier の数値）、`docs/report.ja.md` /
  `docs/report.md`（32B v2 節）、`README.md`（Findings 8）、`.gitignore`（JevBench の server.log を追跡しない）、`results/public/jevbench_targeted_request.md`（提出文）。
  結果: `results/jevbench/qwen3-32b_hardfam_v2_T/{easy,standard,hard}`、`results/jevbench_families_32b_v2.json`、`results/compare_{mmlu,jmmlu}_32b_v2.json`、
  `results/compare_synth_*_32b_v2.json`、`results/synth_difficulty_32b_v2.md`、`results/*_slot_qwen3-32b_32b-hardfam-v2*`、`results/train_32b-hardfam-v2_log.jsonl`。
  checkpoint（adapter + head + temperature、148 MB）は GitHub release `targeted-v2-32b`（SHA-256 fbdcb2da…6303）。
- Verified: 32B v2 head で JevBench public hard 82 / 111 = 0.739（zero-shot 68、v1 head 72。v2 vs zero-shot 19 勝 5 敗 p=0.0066、vs v1 15 勝 5 敗 p=0.041）、
  temporal_numeric 7 / 15、probability 9 / 10、long_policy 11 / 19、easy 1.000、standard 0.958、served ECE 0.118、Brier 0.390。synth temporal_v2 test 0.716
  （zero-shot 0.184、v1 head 0.428）、MMLU 0.814（v1 0.821）、JMMLU 0.786（v1 0.779）。GB10 A で `/health` が engine slot、prompt_hash 4f85a0b34776、T 1.8190 を返すことを確認。
  release の asset URL が 200 を返し、sha256 ファイルの値が一致。
- Deviations: gate 5 条件のうち ECE のみ v1 より悪い（0.096 → 0.118。zero-shot 0.127 よりは良い）。ユーザーの判断で提出した。
  JevBench は提出時点で v1.4.0（sealed 308 問、public-to-sealed gap の罰則）に移行しており、jqv の v1.4 row は "jqv (Qwen3-32B zero-shot)" score 44.35（ranked）。
  提出文には、生成器の scenario を public hard の誤答を読んで設計したこと（public hard は開発 gate、学習データは合成で contamination 0）を明記した。
  以前の run の `server.log` 22 件は既に追跡済みのまま（今回の分だけ未追跡にした）。
- Remaining: Benchmark Heaven の測定待ち（jevbench#51、監視中）。held-out / sealed の結果が出たら report に追記する。ECE は決定型の校正集合で温度を当て直す余地がある
  （`correctness-calibrator` / `proper-scoring-true-prob` の draft）。30 か月上限・金利期間の日割りなど v2 が覆わない計算型が残る。
