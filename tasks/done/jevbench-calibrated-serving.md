---
title: JevBench を校正済み確率（温度ファイルをサーバに適用）で 9 設定実行し、ハーネス側で測った ECE を README に反映する
status: done
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

# Result

## Changed

- `scripts/jevbench_run.py --temperature-file`（config.json に記録）、`scripts/jevbench_summary.py`（served-T run は事後スケーリングしない）
- `jqv/systemone.py`: サーバに温度があるときは校正済み確率を返す（従来は温度を渡しても生確率を返していたバグを修正。`tests/test_systemone.py` に 1 件追加）
- `results/jevbench/qwen3-{1.7b,14b,32b}_{packed,permavg,slot}_T/`（9 設定 × 3 tier、失敗 0）、`results/jevbench_summary.{md,json}`
- `README.md`: JevBench 表を served-T の 9 行に差し替え、測定方法と結論 4 を更新

## Verified

- 9 設定すべて 48/48・72/72・111/111 で失敗 0。精度は生確率 run と同一（温度は argmax 不変）
- ハーネス計測の hard ECE（served-T）は事後適用値と一致（32B zero-shot 0.107 / perm_avg 0.115 / slot 0.132、14B 0.126 / 0.185 / 0.193、1.7B 0.151 / 0.139 / 0.146）
- `uv run pytest`（systemone 6 件を含む）

## Deviations

- 最初の起動時に `/v1/systemone` が生確率を返すバグに気づいて修正・再起動した際、親の `sh -c` チェーンを止め損ねて 2 本が同時に走った。
  すべて停止して競合した run（1.7B perm_avg_T）を削除し、1 本のチェーンで再実行した。以後のチェーンは `: JEVCHAIN_T2` のマーカーで pkill できるようにした
- 1.7B の served-T ECE は事後適用値と 0.003〜0.006 の差（bf16 の run 間ばらつき）

## Remaining

- held-out / judge tier（公開 endpoint + bench request）
