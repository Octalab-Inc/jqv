---
title: temporal_numeric 生成器 v2: JevBench で落とす計算型を教師データにして負の転移をなくす
status: done
priority: P1
created_at: 2026-09-23T00:23:31+09:00
depends_on:
  - hard-family-synth-data
---

# Goal

temporal_numeric の合成生成器（`jqv/synth/temporal.py`）を、JevBench hard の temporal 問題で実際に落としている計算型に合わせて作り直し、
同じ recipe（slot + LoRA、synth 70% + MMLU 30% の混合）で学習した 14B が JevBench public hard の temporal_numeric で zero-shot より悪くならない
（負の転移が消える）ことを確認する。

# Context

- `hard-family-targeted-lora` の結果: synth temporal test は 14B +42 pt、32B +37 pt（p<0.001）なのに、JevBench public hard の temporal_numeric は
  14B 5 → 3 / 15、32B 4 → 2 / 15 と 2 サイズで再現する負の転移。他 family は維持か改善（`docs/report.ja.md` の 32B 節）。
- 32B で落とした 3 問: EUR の金額計算（`eur_498_81` → `eur_513_69`）、30 か月上限の失効判定（`expired_30_month_cap` → `covered`）、期限の yes/no。
  いずれも「選択肢を決めるまでの数値・日付計算」を要する。
- 現在の生成器の scenario: `warranty_month_end_tz`、`business_day_deadline`、`service_months_band`、`prorated_invoice`、`unit_threshold`、`dst_cutoff`
  （月末規則、営業日、DST、日割り、単位換算が中心）。答えは標準ライブラリ（datetime / zoneinfo / decimal）で検証し、Trace から `dependency_hops` /
  `reasoning_depth` を付け、誤誘導メモ（distractor）を入れる（`jqv/synth/common.py`）。train は `python -m jqv.synth.generate` で seed 0 から再生成し、
  `train.paraphrase.jsonl` の patch を再適用する。dev / test は git 管理。
- JevBench public との contamination check（8-gram 共有 0）は `scripts/synth_contamination.py --jevbench <harness>/datasets/public`。
  難易度は `scripts/synth_difficulty.py --model Qwen/Qwen3-32B --split dev` で zero-shot 精度を見る（現行 family は 0.3〜0.5）。
- 学習・評価の手順は `hard-family-targeted-lora` と同じ（`scripts/train_head.py --train-mix ...`、`scripts/eval.py --head-dir`、`scripts/compare_runs.py`、
  `scripts/jevbench_run.py --tiers hard`、`scripts/jevbench_families.py`）。14B は Mac で 1 run 約 10 時間、GB10（`docs/gb10.md`）なら短い。
- JevBench の temporal_numeric は public 15 問しかない。合否の判定には使えないので、方向（zero-shot を下回らない）と、落としていた問題型の回復を見る。

# Scope

## In

- `jqv/synth/temporal.py` に「choice を決めるまでの数値計算」を教師データにする scenario を追加する:
  契約期間 + 上限期間（有効期間と cap のどちらが先に尽きるか）、金額 + 通貨 + 閾値（換算・丸め・税込みの後に閾値と比較）、
  発効日 + 失効条件（複数の失効条件のうち最初に成立するもの）、期限以前 / 以後の boolean 判定、複数数値条件の AND / OR。
  各 scenario は program-verified な答え、Trace（`dependency_hops`）、誤誘導メモ、choice / noul / score の型を持つ。
- dev / test の再生成と paraphrase patch の再適用、contamination check（8-gram 共有 0）、32B zero-shot の難易度確認（v1 と同程度に難しいこと）。
- test は文章表現だけでなく rule / program seed 単位で train から分離する（同じ計算問題の言い換えが test に入らない。program key の hash で split を決める）。
- 14B で同じ recipe で学習し、synth test（v2）、MMLU test 800、JevBench public hard（family 別）を v1 の 14B run（`results/jevbench/qwen3-14b_hardfam_T`）と比較する。
- `docs/report.ja.md` に結果を記録する（v1 との対比、落としていた問題型が戻ったか）。

## Out

- prompt 形式の変更（T と prompt_hash が変わる）。
- long_policy / probability の生成器の変更（共有ユーティリティの修正は可）。
- 32B での学習（14B の結果を見てから別に判断する）。
- RLCD などの学習法の変更。

# Success

- v2 の temporal_numeric test（500）で 14B の targeted LoRA が zero-shot 比 +10 pt 以上（McNemar p < 0.01）。
- JevBench public hard の temporal_numeric が 14B で zero-shot（5 / 15）以上、かつ他 family の合計が v1 の 14B run（67 / 111）から 2 問以上落ちない。
- 32B で落とした 3 問の型（金額計算、期間上限、期限前後の判定）が v2 の scenario で各 1 つ以上覆われている。
- MMLU test 800 が zero-shot 比 −1.0 pt 以内。
- contamination check が 8-gram 共有 0。
- v2 の train / dev / test の間で program key（scenario + 解かれた facts）の重複が 0。

# Verify

```bash
uv run pytest tests/test_synth.py
uv run python -m jqv.synth.generate --family temporal_numeric
uv run python scripts/synth_contamination.py --jevbench /Users/h.imura/tmp/repo/jevbench/datasets/public
uv run python scripts/synth_difficulty.py --model Qwen/Qwen3-32B --split dev
uv run python scripts/train_head.py --run-name 14b-hardfam-v2 --model Qwen/Qwen3-14B --head slot --lora-rank 16 \
  --train-mix synth:long_policy=0.25,synth:temporal_numeric=0.25,synth:probability=0.2,mmlu=0.3 \
  --val-mix synth:long_policy:dev=96,synth:temporal_numeric:dev=96,synth:probability:dev=96,mmlu_val=64 \
  --steps 600 --batch-size 8 --grad-accum 2 --grad-checkpointing --max-len 4096 --eval-every 100 --ckpt-every 100 --brier-weight 0
uv run python scripts/eval.py --dataset synth:temporal_numeric:test --n 0 --n-val 0 --model Qwen/Qwen3-14B --engine slot --head-dir results/train/14b-hardfam-v2/best
uv run python scripts/compare_runs.py --dataset synth-temporal_numeric-test --runs "zero-shot=packed_qwen3-14b" "v2=slot_qwen3-14b_14b-hardfam-v2" --out compare_synth_temporal_numeric_14b_v2.json
uv run python scripts/jevbench_run.py --model Qwen/Qwen3-14B --engine slot --head-dir results/train/14b-hardfam-v2/best --temperature-file results/mmlu_slot_qwen3-14b_14b-hardfam-v2_temperature.json --label qwen3-14b_hardfam_v2_T --tiers hard
uv run python scripts/jevbench_families.py --labels qwen3-14b_packed_T qwen3-14b_hardfam_T qwen3-14b_hardfam_v2_T
```

# Decisions（2026-09-23、ユーザー）

- v1 scenario は残す（v2 だけに置換すると、月末・営業日・DST・単位換算の能力を忘れたのか新しい計算型が効いたのか分からなくなる）。
- temporal の混合比は v1 40% / v2 60% で開始。v2 の 5 種類は原則均等。
- 14B の学習は GB10 で回す（A = 学習、B = 評価と zero-shot 比較）。Mac との数値差を見る実験ではない。
- test は rule / program seed 単位で train から分離する。
- Success gate は予定どおり: 14B で v2 synth が v1 より改善し、JevBench temporal_numeric が zero-shot を下回らないこと。全 hard と MMLU は非退行の確認に使う。
- 目的は学習量を増やすことではなく、JevBench で要求される計算型へ学習分布を合わせると負の転移が消えるかの検証。通れば 32B へ。

# Result

- Changed: `jqv/synth/temporal_v2.py`（新 family `temporal_v2`、5 scenario、program key を signature にする split）、`jqv/synth/__init__.py`（FAMILIES）、
  `jqv/synth/generate.py`（MODULES）、`jqv/synth/paraphrase.py`（CUDA 対応、v2 の事実段落の接頭辞）、`tests/test_synth.py`（solver 5 本 + split 分離 + 妥当性）、
  `data/synth/temporal_v2/{dev,test}.jsonl`、`summary.json`、`train.paraphrase.jsonl`（400 item）。結果: `results/jevbench/qwen3-14b_hardfam_v2_T`、
  `results/jevbench_families_14b_v2.json`、`results/compare_synth_*_14b_v2.json`、`results/compare_mmlu_14b_v2.json`、`results/synth_difficulty_14b_v2.md`、
  `results/synth-temporal_v2-*`、`results/train_14b-hardfam-v2_log.jsonl`。checkpoint は `results/train/14b-hardfam-v2/best`（git 外、Mac に回収済み）。
  `docs/report.ja.md` の temporal_v2 節（設計 + 結果）、`docs/report.md` の英語節、README の Findings。
- Verified: gate 4 条件とも通過。temporal_v2 test 0.244 → 0.614（v1 head 0.358、p<0.001）；JevBench temporal_numeric 6/15（zero-shot 5、v1 head 3）；
  hard 全体 61 → 74 / 111（v1 run 67。v2 vs zero-shot 22 勝 9 敗 p=0.029）；MMLU 0.750 → 0.764。contamination 8-gram 共有 0、program key の split 重複 0、
  train の md5 が Mac と GB10 で一致（seed 0 + patch から再現可能）。
- Deviations: 言い換えの対象は train の 20%（479 候補段落 → 400 採用。v1 は 28%）。paraphrase.py が GB10 では CPU で動いていた（MPS/CPU しか見ていなかった）ため、
  初回は学習前に止めて CUDA 対応後に再実行した。v1 temporal の synth test は v1 head 比 −14 pt（temporal 枠 0.25 → 0.10 の分。JevBench には出ない）。
  judge_hard が v1 run から 2 問減（合計では +7）。v2 synth の zero-shot 難易度は v1 より高い（32B 0.184）。
- Remaining: 手本にした 2 問（EUR 換算、30 か月上限）は v2 でも不正解。fx_lines_cap は scenario 中最も低い（0.48）。32B の学習は別判断（ユーザー）。
- Conclusion: matching the temporal training distribution to the computations JevBench asks for removes the negative transfer at 14B (3 → 6 / 15) and lifts
  the whole public hard tier to 74 / 111, the first paired improvement that is significant (p=0.029); the effect is computation-type generalisation, not
  item memorisation (contamination 0, the two modelled items still fail).
