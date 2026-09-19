---
title: CE + λ·Brier の校正指向学習で ECE が temperature scaling を超えて下がるか検証する
status: pending
priority: P3
created_at: 2026-09-20T01:50:47+09:00
depends_on:
  - pointer-head-lora
---

# Goal

`L = CE + λ·Brier` で decision head（＋LoRA）を学習し、in-distribution と transfer の両方で ECE が post-hoc temperature scaling 単独より下がるか、Jev の ECE 0.031（MMLU 1,200 問, 10-bin）に届くかを検証する。

# Context

- 現状: temperature scaling で MMLU ECE 0.41→0.08、JMMLU 0.48→0.06。Jev は zero-shot で 0.031。
- Brier は proper scoring rule なので、CE に加えることで確率の形自体を学習目標にできる。
- `pointer-head-lora` で作る学習パイプライン（`jqv/train/`, `scripts/train_head.py`）を前提とする。
- `temperature-transfer` の表（あれば）が「学習なし」の基準になる。

# Scope

## In

- `jqv/train/`: loss に `--brier-weight λ` を追加
- λ ∈ {0, 0.5, 1, 2} の sweep（学習設定は `pointer-head-lora` と同じ）
- 評価: MMLU / JMMLU / bridge の test で accuracy, ECE, NLL, Brier を、(a) 学習後そのまま (b) さらに post-hoc T を当てた場合、の両方で記録
- README: λ ごとの表と結論（ECE が下がるか、accuracy とのトレードオフ、Jev との距離）

## Out

- RL / preference optimization（結果次第で次タスクとして起こす）

# Success

- README に λ 4 水準 × データセット 3 の表（ECE, NLL, Brier, accuracy）が (a)(b) 両方である
- 「校正指向学習は temperature scaling 単独より ECE を下げる／下げない」の結論が数値付きで書かれている
- 各学習が進捗ログと checkpoint 付きで実行され、中断後に再開できる

# Verify

```bash
uv run scripts/train_head.py --head pointer --brier-weight 1.0
uv run scripts/eval.py --dataset mmlu --engine pointer --n 1200 --n-val 400
```

コマンドの詳細は `pointer-head-lora` の実装後に確定する。
