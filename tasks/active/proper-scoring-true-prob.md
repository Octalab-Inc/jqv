---
title: 真の確率が分かる合成データで CE / Brier / proper scoring の学習目標を比較し、RLCD に進むか決める
status: draft
priority: P3
created_at: 2026-09-21T10:11:14+09:00
depends_on:
  - hard-family-synth-data
---
# Goal

真の確率分布が分かるデータ（合成 probability family と simulator 型データ）で、学習目標（観測結果への CE、argmax への CE、真の分布に対する Brier、
真の分布に対する log score、CE + λ·Brier）を比較し、分布への忠実度（TVD / KL）と ECE でどれが良いかを示す。その結果から RLCD に進むかを決める。

# Context

- E 節の結果: CE + λ·Brier は λ ≤ 1 で CE と区別がつかず、λ=2 は分布を平坦にした。温度スケーリングを超えていない。
- NanoJev の整理: logits と真値が取れるなら Brier を直接 backprop するのが自然で、RL は sampling / interaction しか得られないときの手段。
- Benchmark Heaven の Calibration 軸は ECE に加えて分布正解問題の probability fidelity を含む（jqv-32B は 71.1）。
- `jqv/train/trainer.py` の `brier_loss` は one-hot 正解のみ。soft target（`target_distribution`）は未対応。
- `hard-family-synth-data` の probability family は `target_distribution` を持つ。

# Scope

## In

- trainer の soft target 対応（`target_distribution` があれば CE / Brier / log score を分布に対して計算）。
- simulator 型データ生成（例: 壺・サイコロ・簡単な待ち行列で真の P が計算できる過程）と、「観測結果 1 つだけをラベルにする」variant
  （サンプルからの CE で分布が復元できるかを見る）。
- 1.7B と 14B、各 2 seed で目標関数 5 種を比較。評価は held-out 合成 test の TVD / KL / ECE、参考として JevBench public probability 10 問。
- README に表と結論（RLCD に進むか、進むならどの設定か）。

## Out

- RLCD の実装。
- 32B の学習。

# Success

- 目標関数 × 指標の表（1.7B / 14B、seed 2 つ、平均と幅）が `results/proper_scoring.md` にある。
- soft target の損失に単体テストがある。
- RLCD の go / no-go が理由付きで README に記録されている。

# Verify

```bash
uv run pytest tests/test_trainer_soft.py
uv run python scripts/train_head.py --run-name 1.7b-prob-brier --objective brier --train-mix synth:probability=1.0 --steps 300 ...   # --objective は候補
uv run python scripts/eval_distribution.py --dataset synth:probability:test --head-dir results/train/1.7b-prob-brier               # 候補
```

# Open Questions

- simulator の過程をどこまで増やすか。既定は 3 種。
