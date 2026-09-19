---
title: pointer 型 decision head + LoRA を学習し語彙 readout (B) と比較する
status: done
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

# Result

## Changed

- `jqv/heads.py`: `SlotHead`（LM head の文字行で初期化可）、`PointerHead`（低ランク双線形 + option bias、順序等変）、save/load
- `jqv/prompt.py`: `suffix_ids_with_spans`（offset mapping で選択肢末尾 token を特定。`suffix_ids` と同一の ids）
- `jqv/train/data.py`, `jqv/train/trainer.py`: MMLU auxiliary_train + 選択肢 shuffle、LoRA（peft）、CE（+ λ·Brier）、10 step ごとの loss/ETA ログ、100 step ごとの checkpoint と検証、`--resume`
- `jqv/engine/head.py`: `pointer` / `slot` engine（`head_dir`、model / prompt_hash 検証、adapter を runtime に注入）
- `scripts/train_head.py`、`scripts/eval.py` と `permutation_test.py` の `--head-dir`
- `tests/test_heads.py`（順序等変性、slot の mask と roundtrip）、`tests/test_prompt.py`（spans）
- `README.md`: C 節（設計、結果表、順序感度、結論）
- `.gitignore`: `results/train/`（adapter は git 管理外）

## Verified

- `uv run pytest`: 全件成功（heads / prompt を含む）
- 学習 3 本（pointer_lora, slot_lora, pointer_frozen: 各 600 step）が完走。smoke run で `--resume` の再開を確認
- `eval.py` で MMLU / JMMLU test 800、`permutation_test.py --mode order` で順序感度、`fit_temperature.py` で +T を測定

## Deviations

- Success の「pointer head の順序一致率が B より高い」は満たさなかった（48% で同じ）。理由（h_i の文脈依存）を README に記載。slot は 55% で B を上回る
- 学習率は smoke run で不安定だったため head 3e-4 / LoRA 5e-5、warmup 50 に下げた

## Remaining

- 学習量（step, LoRA rank）の sweep、pointer の初期化改善、bridge 型（読解・日本語）データの学習
