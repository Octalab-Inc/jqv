---
title: temperature がデータセット・言語・ドメイン間でどこまで転移するかを示す
status: pending
priority: P2
created_at: 2026-09-20T01:50:42+09:00
depends_on: []
---

# Goal

あるデータセットで学習した temperature を別のデータセット・言語・ドメインに適用したときの ECE / NLL / Brier を表にし、「scalar 校正はタスク間で転移するか」に答える。Jev の ECE 0.031 は zero-shot 値なので、こちらが本当の比較対象になる。

# Context

- `results/mmlu_packed_qwen3-1.7b.npz`, `results/jmmlu_packed_qwen3-1.7b.npz`（各 val 400 / test 800）, `results/bridge_packed_qwen3-1.7b.npz`（test 30、val なし）に logits / labels / is_val が保存済み。再推論は不要。
- in-distribution では T(MMLU)=11.9 で ECE 0.41→0.08、T(JMMLU)=12.8 で 0.48→0.06。二つの T が近いことは「ほぼ一定の scale distortion」の可能性を示唆する。
- `jqv/calibration.py` の `TemperatureScaler`、`jqv/metrics.py` の `summary` を使う。

# Scope

## In

- `scripts/transfer_temperature.py`: source ∈ {mmlu, jmmlu} の val で T を学習し、target ∈ {mmlu, jmmlu, bridge} の test に適用。対角（in-distribution）と非対角（transfer）を同じ表に出す。T=1（校正なし）の行も含める
- 出力: `results/transfer_qwen3-1.7b.json` と Markdown 表、target ごとの reliability diagram（transfer 時）
- README に「temperature の転移」節を追加し、結論を 2〜3 文で書く

## Out

- 新しいデータセットの追加、モデルの変更
- temperature 以外の校正手法

# Success

- README に source 2 × target 3 の表（T, ECE, NLL, Brier）と T=1 の基準行がある
- 各セルの n が明記されている（MMLU/JMMLU test 800、bridge 30）
- 「T は転移する／しない」の結論が数値付きで書かれている（例: 非対角の ECE が対角より X 以内なら転移するとみなす。閾値は結果を見て明示する）

# Verify

```bash
uv run scripts/transfer_temperature.py
cat results/transfer_qwen3-1.7b.json
```
