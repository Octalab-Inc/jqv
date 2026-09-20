---
title: JevBench を校正済み確率（温度ファイルをサーバに適用）で 9 設定実行し、ハーネス側で測った ECE を README に反映する
status: pending
priority: P2
created_at: 2026-09-21T04:02:12+09:00
depends_on: []
---

# Goal

JevBench public 231 決定を、jqv サーバが MMLU val で学習した温度を適用した確率を返す「出荷構成」で 9 設定（1.7B / 14B / 32B × zero-shot / perm_avg / slot+LoRA）
実行し、ハーネス自身が測った ECE・Calibration を README の表にする（前回は生の確率を返し、温度は集計側で事後適用していた）。

# Context

- `scripts/jevbench_run.py` はサーバ起動時に温度ファイルを渡せない。`jqv/server.py` は `JQV_TEMPERATURE_FILE` で温度を適用し、model / prompt_hash の provenance を検証する。
- 温度ファイル: packed → `results/mmlu_packed_<slug>_temperature.json`、perm_avg → `..._permavg_temperature.json`、slot → `results/mmlu_slot_<slug>_<run>_temperature.json`。
- 前回の結果は `results/jevbench/<label>/`（生確率）。今回は `<label>_T` に保存し、集計では温度の二重適用を避ける。

# Scope

## In

- `scripts/jevbench_run.py --temperature-file`（config.json に記録）
- `scripts/jevbench_summary.py`: temperature_file 付き run は事後スケーリングせず、ハーネスの ECE をそのまま「+T」列に出す
- 9 設定の実行と集計、README の JevBench 表の差し替え（生の run と校正済み run の両方を残す）

## Out

- held-out / judge tier、新しい学習

# Success

- `results/jevbench/qwen3-{1.7b,14b,32b}_{packed,permavg,slot}_T/` の 3 tier が全件完走（失敗 0）
- README の JevBench 表に、サーバが返した校正済み確率でハーネスが測った hard ECE と Calibration がある
- 校正済み run の精度が生 run と一致する（温度は argmax を変えない）

# Verify

```bash
uv run scripts/jevbench_run.py --model Qwen/Qwen3-32B --engine packed --temperature-file results/mmlu_packed_qwen3-32b_temperature.json --label qwen3-32b_packed_T
uv run scripts/jevbench_summary.py
```
