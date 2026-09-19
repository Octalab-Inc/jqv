---
title: Qwen3-32B で最良設定だけを検証し backbone スケーリング表（Jev 比較）を完成させる
status: pending
priority: P3
created_at: 2026-09-20T05:14:26+09:00
depends_on:
  - scaling-14b-lora-sweep
---

# Goal

Qwen3-32B で語彙 readout（B）と、14B で選んだ最良設定の slot + LoRA を 1 本だけ学習・評価し、
backbone スケーリング表（1.7B / 14B / 32B / Jev）を完成させる。「Jev の MMLU 0.918 のうち backbone 能力で説明できる部分」と
「decision-specific post-training に残る部分」を README で結論づける。

# Context

- Qwen3-32B は BF16 で 65.5 GB（Apache-2.0）。推論は余裕があるが、LoRA 学習は活性化を含めると 128 GB に対して余裕が少ない。
  `scaling-14b-lora-sweep` で追加する gradient checkpointing と勾配累積（micro batch 2〜4）を使う。
- 演算量は 1.7B の約 19 倍。学習 1 本は **6〜8 時間**、評価は 1 本 1 時間程度の見込み。naive の bench は S=2000・Q=10 までに限定する。
- λ sweep は 32B では行わない（14B の最良設定のみ）。

# Scope

## In

- `scripts/eval.py --model Qwen/Qwen3-32B --engine packed`（MMLU / JMMLU / bridge）、`fit_temperature.py`、`transfer_temperature.py --model-slug qwen3-32b`
- `scripts/bench.py --model Qwen/Qwen3-32B --engines kvcache packed shared --state-tokens 2000 8000 --questions 100`（+ S=2000 で naive Q=10）
- `scripts/train_head.py --model Qwen/Qwen3-32B --head slot --brier-weight <best>`（`results/scaling_best_config.json` の値）、
  `--grad-checkpointing --grad-accum` を使用。評価は MMLU / JMMLU / bridge、`compare_runs.py`
- `scripts/scaling_table.py` で最終表。README に「Backbone スケーリング」の結論（精度・ECE の backbone 依存、Jev との残差、
  スループットの backbone 依存）を書く
- メモリが足りない場合は micro batch 1 + 勾配累積 8、それでも足りなければ LoRA 対象を q/v_proj に減らす。変更は README に記録

## Out

- 32B での λ sweep、pointer head、MoE / Qwen3.5 モデル
- 4-bit / 8-bit 量子化での学習（bf16 で収まらない場合のみ Open Questions として起こす）

# Success

- README のスケーリング表に 32B の行（raw B、slot+LoRA、ECE raw / +T、packed q/s）が入り、Jev 行（0.918 / 0.031、Hume の測定値と明記）と並ぶ
- 「backbone を 1.7B → 14B → 32B にしたとき MMLU 精度と ECE がどう動いたか」と「32B でも Jev に届かない差の大きさ」が数値付きで書かれている
- `results/train/qwen3-32b_slot_best/best` と対応する評価 JSON が存在する
- 学習が 128 GB 内で完走した設定（micro batch、勾配累積、checkpointing の有無）が README に記録されている

# Verify

```bash
uv run scripts/eval.py --model Qwen/Qwen3-32B --dataset mmlu --engine packed --n 1200 --n-val 400
uv run python -u scripts/train_head.py --model Qwen/Qwen3-32B --head slot --run-name qwen3-32b_slot_best --steps 600 --brier-weight <best> --grad-checkpointing --grad-accum 4 --batch-size 2
uv run scripts/eval.py --model Qwen/Qwen3-32B --dataset mmlu --engine slot --head-dir results/train/qwen3-32b_slot_best/best --n 1200 --n-val 400
uv run scripts/scaling_table.py
```

# Open Questions

- bf16 で学習が 128 GB に収まらない場合、量子化学習（QLoRA）に進むかどうか
