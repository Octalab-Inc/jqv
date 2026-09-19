---
title: pointer 型 decision head + LoRA を学習し語彙 readout (B) と比較する
status: pending
priority: P3
created_at: 2026-09-20T01:50:46+09:00
depends_on: []
---

# Goal

語彙 head に依存しない decision head を学習し、語彙 readout (B) と同じ評価で精度・校正・順序頑健性を比較する。head は pointer 型（decision 表現 h_d と各選択肢表現 h_i の双線形スコア z_i = h_dᵀ W h_i）を主とし、slot 型（`Linear(d, max_choices)`、LLM freeze）を比較用に置く。

# Context

- 現状の engine は全て語彙 readout。README「次フェーズ」で C として予定していた。
- Hume は Jev の readout として slot head と pointer scorer の両方を候補に挙げている。pointer 型は K 可変・順序に自然・任意ラベルに強い。
- slot head を LLM freeze で新規初期化すると、hidden state のどこに「option i の正しさ」が乗っているかを LLM が学習していないため B より悪化する可能性がある。LoRA を併用する。
- 学習データ: MMLU `auxiliary_train`（99,842 問）と `dev`。JMMLU は test のみなので評価専用。`peft` は未導入。
- 選択肢表現 h_i の取り方（各選択肢行の末尾 token など）は本タスクで決めて README に書く。
- 学習は MPS で行う。進捗の可視化（step ごとの loss、ETA、checkpoint）と再開を必須にする（過去にベンチが可視化なしで走り結果を失った経緯がある）。

# Scope

## In

- `jqv/engine/pointer.py`, `jqv/engine/slot.py`: 学習済み head で decide する engine
- `jqv/train/`: データ整形（選択肢順序の shuffle 拡張を含む）、loss（CE）、LoRA（peft）、checkpoint 保存と再開
- `scripts/train_head.py`: `--head pointer|slot`, `--lora-rank`, `--max-steps`, `--resume`。step ごとに loss と ETA をログに出す
- `scripts/eval.py`: 新 engine を選べるようにする
- README: C1/C2 の設計、学習設定、B との比較表（MMLU/JMMLU test 800: accuracy, ECE, NLL; option-order permutation に対する argmax 一致率）

## Out

- RL / preference optimization
- Brier 項の追加（`calibration-training`）

# Success

- pointer head + LoRA の MMLU test 800 問の accuracy が B 以上、または下回る場合はその差と考えられる理由が README にある
- pointer と slot の比較（accuracy, ECE）が README にある
- option-order permutation の argmax 一致率が pointer head で B より高い、または差がないことが示されている
- `--max-steps 20` の学習が途中で中断しても `--resume` で続きから再開できる
- `uv run pytest` が全件成功する

# Verify

```bash
uv run pytest
uv run scripts/train_head.py --head pointer --max-steps 20
uv run scripts/eval.py --dataset mmlu --engine pointer --n 1200 --n-val 400
```

コマンドの詳細は本タスクで確定する。

# Open Questions

- 学習ステップ数と LoRA rank。MPS での学習時間の上限をどこに置くか（例: 1 設定 1 時間以内）
- h_i を取る位置（選択肢行末の改行 token か、選択肢テキスト末尾か）
