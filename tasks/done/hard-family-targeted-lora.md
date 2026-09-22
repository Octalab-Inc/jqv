---
title: 合成データによる targeted LoRA で弱い family の精度を上げる（14B で判定し、効けば 32B）
status: done
priority: P1
created_at: 2026-09-21T10:11:14+09:00
depends_on:
  - hard-family-synth-data
---
# Goal

合成データ（`hard-family-synth-data`）で slot head + LoRA を学習し、long_policy / temporal_numeric / probability の精度を上げる。
14B で効果を判定し（gate）、通った場合だけ 32B で最終学習と JevBench public hard の再測定を行う。効かなかった場合は
負の結果として記録して終える（32B を無理に回さない）。

# Context

- これまでの結果: MMLU train 4,800 例の slot + LoRA は 1.7B で +2〜4 pt、14B / 32B ではほぼ無効（README C 節、Backbone スケーリング節）。
  CE + λ·Brier は λ ≤ 1 で CE と区別がつかず、14B の最良は λ=0。
- 学習器: `scripts/train_head.py`（`--head slot --lora-rank 16 --steps --batch-size --brier-weight --max-len --grad-accum --grad-checkpointing --resume --model`）。
  データは `jqv/train/data.py:load_split` が `cais/mmlu` を読む固定実装で、JSONL 入力と複数ソースの混合は未対応。選択肢 shuffle はある。
- 速度: 14B は 0.10 step/s（batch 8、`--grad-accum 2`、MMLU 長）で 600 step ≈ 100 分。32B は 0.03 step/s（`--grad-checkpointing --grad-accum 4`）。
  long_policy は 1 例 2〜4k token なので step 時間は数倍になる見込み。最初の 20 step で実測して ETA を出す。
- 評価: `scripts/eval.py --head-dir`、`scripts/compare_runs.py`（McNemar + bootstrap + 選択的精度、ラベルに "=" 不可）、
  `scripts/jevbench_run.py`（served-T 構成、`--force` は tier 単位）、`scripts/jevbench_families.py`（前タスク）。
- JevBench public hard は n=111（標準誤差 ≈ 4.6 pt）、family は 10〜19 問なので、合否の gate には使わず指標として CI 付きで報告する。
- メモリは 128 GB の上限付近で動く。学習中に別の GPU ジョブを走らせない。

# Scope

## In

- `jqv/train/data.py` の拡張: JSONL（`data/synth/...`）の読み込み、複数ソースの重み付き混合（例: synth 3 family 70% + MMLU train 30% の replay）、
  `--max-len 4096`、長さでバケット化した batch（step 時間の安定化）。`--train-mix` のような指定方法を追加する。
- 14B の学習: slot + LoRA r=16、λ=0、600〜1,000 step × batch 8、seed 固定。最初の 20 step で step 時間を測って ETA を報告し、
  100 step ごとの val（synth dev の混合）を出す。checkpoint と `--resume` を使う。
- 評価（14B）: synth test（family 別と `dependency_hops` 別、各 500 問）、MMLU test 800（T は val 400 で再学習）、JevBench public hard（family 別 + 全体、bootstrap CI）。
  比較は zero-shot 14B との対応比較（McNemar）。
- gate を通った場合の 32B: 同じ設定で学習し、同じ評価 + JMMLU 800 を行い、README の Findings / Backbone / JevBench 節を更新する。
- gate に落ちた場合: 負の結果（設定、学習曲線、family 別の数値）を README に記録して終了する。

## Out

- RL、全パラメータ fine-tuning、新しい head 構造。
- prompt 形式の変更（T と prompt_hash が変わる）。
- multi_hop / tradeoff への拡張。
- perm_avg との組み合わせ（評価の参考値としてのみ 1 回）。

# Success

- 14B gate: synth test で 3 family 中 2 family 以上が zero-shot 比 +10 pt 以上（McNemar p < 0.01）、かつ MMLU test 800 が −1.0 pt 以内。
- gate 通過時の 32B: 同じ基準を満たし、JevBench public hard の family 別・全体を CI 付きで報告する（全体の改善は指標であり合否条件ではない）。
- どちらの場合も `results/train/<run>/` に学習ログ・checkpoint、`results/hardfam_<model>.md` に評価表、README に結論がある。
- 学習と評価は進捗行と ETA を出して実行され、途中結果は逐次保存される。

# Verify

```bash
uv run python scripts/train_head.py --run-name 14b-hardfam --model Qwen/Qwen3-14B --head slot --lora-rank 16 \
  --train-mix synth:long_policy=0.25,synth:temporal_numeric=0.25,synth:probability=0.2,mmlu=0.3 \
  --steps 600 --batch-size 8 --grad-accum 2 --grad-checkpointing --max-len 4096      # --train-mix は候補。実装後に確定
uv run python scripts/eval.py --dataset synth:long_policy:test --model Qwen/Qwen3-14B --engine packed --head-dir results/train/14b-hardfam
uv run python scripts/compare_runs.py --a results/synth_long_policy_test_qwen3-14b.json --b results/synth_long_policy_test_qwen3-14b_hardfam.json   # 候補
uv run python scripts/eval.py --dataset mmlu --model Qwen/Qwen3-14B --engine packed --head-dir results/train/14b-hardfam
uv run python scripts/jevbench_run.py --model Qwen/Qwen3-14B --head-dir results/train/14b-hardfam --temperature-file ...   # --head-dir 対応の有無を確認
uv run python scripts/jevbench_families.py
```

# Open Questions

- 混合比（synth 70% / MMLU 30%）と step 数（600 か 1,000 か）。既定は上記。
- gate の閾値 +10 pt でよいか（synth test n=500 なら ±4 pt の CI）。

# Result

- Changed: `jqv/train/data.py`（JSONL ソース `synth:<family>:<split>`、`--train-mix` / `--val-mix` の重み付き混合、`MixSampler`）、`jqv/train/trainer.py`
  （ソース別 val、sampler cursor の保存と再開）、`scripts/train_head.py`、`scripts/synth_difficulty.py`、`scripts/jevbench_families.py`、`docs/report.ja.md`
  （14B gate 節、32B 節）、`docs/gb10.md`。結果: `results/train_14b-hardfam_log.jsonl`、`results/train_32b-hardfam-gb10_log.jsonl`、`results/compare_synth_*`、
  `results/compare_mmlu_32b_hardfam.json`、`results/compare_jmmlu_32b_hardfam.json`、`results/jevbench/qwen3-14b_hardfam_T`、`results/jevbench/qwen3-32b_hardfam_T`、
  `results/jevbench/qwen3-32b_packed_T_gb10`、`results/jevbench_families_32b_hardfam.json`、`results/synth_difficulty_32b_hardfam.md`。
  checkpoint は `results/train/{14b-hardfam,32b-hardfam-gb10}/best`（git 外、Mac に回収済み）。
- Verified: 14B gate 通過（synth test 3 family +21 / +42 / +27 pt、p<0.001、MMLU +1.0）。32B（GB10、学習 501 分）: synth test +21.6 / +36.6 / +29.8 pt（p<0.001）、
  MMLU 0.806 → 0.821（p=0.058）、JMMLU 0.767 → 0.779（p=0.20）、JevBench public hard 68 → 72 / 111（13 勝 9 敗、exact McNemar p=0.52）、
  served ECE 0.127 → 0.096、Brier 0.516 → 0.415、ordinal MAE 0.77 → 0.56。family 別: probability 4 → 8 / 10、long_policy 9 → 10 / 19、temporal_numeric 4 → 2 / 15。
- Deviations: 評価表は `results/hardfam_<model>.md` ではなく `docs/report.ja.md` の節と `results/compare_*.json` に置いた（レポートは docs/report.ja.md に集約する指示）。
  32B は Mac ではなく GB10 で学習した（`gb10-training-infra`。micro 2 × 累積 4 で 121 GB に収まった）。zero-shot 基準は GB10 で再測定した `*_gb10` を使い、
  Mac の既存値は参考（JevBench は 1 問だけ argmax が反転する、上位 2 択がほぼ同率の問題）。JevBench hard の family 別は CI ではなく exact McNemar と勝敗数で報告した。
- Remaining: temporal_numeric の JevBench での負の転移（14B 5 → 3、32B 4 → 2）は `temporal-generator-v2` へ。probability は JevBench n=10 なので独立 test を増やす。
  long_policy の 6-hop（サブリミット）は 2 サイズとも悪化（0.60 → 0.33、0.57 → 0.27）。
- Conclusion: Targeted LoRA improves in-distribution hard-family performance and probability quality, but transfer to JevBench is heterogeneous.
  Probability reasoning transfers positively, long-policy transfer is limited, and temporal-numeric training shows reproducible negative transfer across
  14B and 32B. Generic targeted fine-tuning is therefore insufficient; the next iteration should redesign the temporal-numeric generator around the actual
  computation patterns observed in held-out-style errors rather than simply increasing data volume.
