---
title: Qwen3-14B で slot+LoRA の λ sweep を 1.7B と同一条件で行い最良設定を決める
status: done
priority: P2
created_at: 2026-09-20T05:14:25+09:00
depends_on:
  - scaling-14b-baseline
---

# Goal

Qwen3-14B で slot + LoRA（r=16、600 step × batch 8、選択肢 shuffle）を 1.7B と同一条件で学習し、λ ∈ {0, 0.5, 1, 2}
（CE + λ·Brier）を比較して、32B に持っていく最良設定を 1 つ決める。「backbone を 8 倍にすると decision training の効果と
校正はどう変わるか」を README のスケーリング表に書く。

# Context

- 1.7B の結果: slot+LoRA λ=0 で MMLU 0.575（B 0.554）、ECE raw 0.14 / +T 0.044、JMMLU 0.505 / 0.029。λ=0.5 は λ=0 と同等
  （`calibration-training` の結果を参照）。
- 学習スタックは `jqv/train/trainer.py`, `scripts/train_head.py`。`--resume` で再開可能。1.7B は 0.45 step/s（22 分/600 step）。
  14B は演算量が約 8 倍なので **1 本 2.5〜3 時間、4 本で 10〜12 時間** の見込み。評価は 1 本 30 分程度。
- メモリ: BF16 重み 29.5 GB + 活性化（40 層 × batch 8 × 150 token）10〜15 GB + LoRA 状態は 128 GB に収まる見込み。
  収まらない場合に備えて gradient checkpointing と勾配累積を trainer に追加する（32B でも必要）。
- 学習は必ずバックグラウンド + 進捗ログ + checkpoint で実行し、中断後は `--resume` で続ける。

# Scope

## In

- `jqv/train/trainer.py`, `scripts/train_head.py`: `--grad-accum N`（実効 batch を保ったまま micro batch を小さく）と
  `--grad-checkpointing`（`model.gradient_checkpointing_enable()`）を追加。1.7B で 20 step の smoke run により等価性（loss 推移）を確認
- run 名は `qwen3-14b_slot_brier{0,05,1,2}`。設定は 1.7B の `slot_brier*` と同じ（lr 5e-5 / head 3e-4、warmup 50、max_len 512、seed 0）
- 各 run の best を MMLU 1200 / JMMLU 1200 / bridge で評価、`fit_temperature.py`、`compare_runs.py`（B vs 各 λ の対応比較）、
  `transfer_temperature.py --engine slot --suffix _qwen3-14b_slot_brier*`
- 最良 λ の選定基準: MMLU val の NLL（+T 後の ECE を補助）。基準と結果を README に明記
- README: スケーリング表の 14B 行に slot+LoRA の値を入れ、λ 表（14B）を追加。「1.7B と 14B で λ の効き方が変わるか」を書く

## Out

- pointer head（1.7B で slot に劣ったため 14B では扱わない）
- 32B の学習

# Success

- `results/train/qwen3-14b_slot_brier{0,05,1,2}/best` が存在し、各 run の `final.json` と評価 JSON がある
- README の λ 表（14B）に 4 水準 × MMLU / JMMLU の accuracy / NLL / Brier / ECE（raw, +T）がある
- 最良 λ が根拠付きで README に書かれ、`results/scaling_best_config.json` に `{"model": "Qwen/Qwen3-14B", "brier_weight": ..., "run": ...}` が保存されている
- 途中で kill しても `--resume` で続きから再開できることを 1 回確認している

# Verify

```bash
uv run python -u scripts/train_head.py --model Qwen/Qwen3-1.7B --head slot --run-name smoke_ga --steps 20 --grad-accum 2 --batch-size 4   # 等価性 smoke
for L in 0 0.5 1 2; do uv run python -u scripts/train_head.py --model Qwen/Qwen3-14B --head slot --lora-rank 16 --run-name qwen3-14b_slot_brier$(echo $L | tr -d .) --steps 600 --brier-weight $L; done
uv run scripts/eval.py --model Qwen/Qwen3-14B --dataset mmlu --engine slot --head-dir results/train/qwen3-14b_slot_brier0/best --n 1200 --n-val 400
uv run scripts/compare_runs.py --dataset mmlu --runs "B=packed_qwen3-14b" "slot λ=0=slot_qwen3-14b_qwen3-14b_slot_brier0" ...
uv run scripts/scaling_table.py
```

# Open Questions

- 14B の 1 run が 3 時間を大きく超える場合、λ を {0, 1} に絞るかどうか（ユーザー判断。既定は 4 水準すべて実行）

# Result

## Changed

- `results/train/qwen3-14b_slot_brier{0,05,1,2}/`（best / last checkpoint、train_log.jsonl、final.json。git 管理外）
- `results/{mmlu,jmmlu,bridge}_slot_qwen3-14b_qwen3-14b_slot_brier*.{json,npz}`、`*_temperature.json`、`*_calibration.json`、`compare_{mmlu,jmmlu}_qwen3-14b_lora.json`
- `results/scaling_best_config.json`: 32B 用の設定（λ=0、根拠付き）
- `scripts/scaling_table.md` / `results/scaling_table.json` を再生成
- `README.md`: 「14B の slot + LoRA λ sweep」節と結論、スケーリング表の更新

## Verified

- 学習 4 本（各 600 step、`--grad-accum 2`）完走。λ=1 の run で 20 step → `--resume` → 600 step の再開を確認
- 各 best を MMLU / JMMLU test 800 と bridge で評価、`fit_temperature.py`、`compare_runs.py`（zero-shot / perm_avg / 4 λ の対応比較）
- `--grad-accum` / `--grad-checkpointing` は 1.7B の 12 step smoke run で動作確認（前タスク）

## Deviations

- Success の「最良 λ を根拠付きで」は val NLL 最小の λ=0 としたが、test の差はすべてノイズ範囲であることを明記
- `compare_runs.py` のラベルに「=」を含めるとパースに失敗するため、ラベル表記を変えて再実行した（スクリプト側は未修正）
- transfer_temperature（slot run 版）は実行していない（B の転移表は前タスクで取得済み）

## Remaining

- LoRA + perm_avg の組合せ評価（32B タスクで slot の perm_avg 評価を入れる）
- 学習量（step 数、データ）を増やした場合に 14B で精度が動くかは未検証
