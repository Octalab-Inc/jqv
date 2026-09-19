---
title: 校正温度の出所を temperature ファイルと API 応答から追跡できるようにする
status: done
priority: P2
created_at: 2026-09-20T01:50:41+09:00
depends_on: []
---

# Goal

校正温度がどのモデル・プロンプト・データセットで学習されたかがファイルと API 応答の両方から分かり、別モデルや別プロンプトへの誤用を起動時に検知できる状態にする。

# Context

- `jqv/calibration.py` の `TemperatureScaler.save` は `{"temperature": T}` しか書かない。
- `jqv/server.py` は `JQV_TEMPERATURE_FILE` を読み、`calibrated_probabilities` を返すが、出所は応答に含まれない。
- README の API 例は MMLU-en で学習した T=11.9 を橋梁点検の日本語文書に流用している。校正は deployment distribution に依存するので、この値を「calibrated」と呼ぶには出所の明示が必要。
- `scripts/fit_temperature.py` が T を学習して保存する。`scripts/eval.py` の npz には dataset / engine / model の情報が JSON 側にある。

# Scope

## In

- `jqv/calibration.py`: save/load にメタデータ（`model`, `engine`, `prompt_hash`（`PromptStyle` と system prompt から算出）, `dataset`, `n_val`, `choice_counts`, `fitted_at`）を持たせる。旧形式（temperature のみ）も読めるようにする
- `scripts/fit_temperature.py`: npz と対応する JSON からメタデータを埋める
- `jqv/server.py`: 起動時に temperature ファイルの `model` / `prompt_hash` が実行時と異なれば例外。`JQV_ALLOW_CALIBRATION_MISMATCH=1` で警告に落とせる
- `jqv/types.py`, `jqv/server.py`: `DecisionResponse.calibration` に `{temperature, source_dataset, model, n_val}` を返す
- README: API 例に「MMLU-en で学習した T の流用であり、この domain で検証していない」と注記し、`calibration` フィールドを説明

## Out

- 校正手法自体の変更（Platt, vector scaling など）
- bridge_synth での T の学習（30 問では不可）

# Success

- `fit_temperature.py` が出力する JSON に上記メタデータが全て入っている
- 旧形式の temperature JSON も `TemperatureScaler.load` で読める
- `/decision` の応答に `calibration` フィールドがあり、temperature 未指定時は `null`
- `JQV_MODEL` を temperature ファイルと異なるモデルにして起動すると失敗し、`JQV_ALLOW_CALIBRATION_MISMATCH=1` なら警告で起動する
- `uv run pytest` が全件成功する

# Verify

```bash
uv run pytest
uv run scripts/fit_temperature.py results/mmlu_packed_qwen3-1.7b.npz
cat results/mmlu_packed_qwen3-1.7b_temperature.json
JQV_DEVICE=cpu JQV_DTYPE=float32 JQV_TEMPERATURE_FILE=results/mmlu_packed_qwen3-1.7b_temperature.json uv run uvicorn jqv.server:app --port 8011
curl -s localhost:8011/decision -H 'content-type: application/json' -d '{"state":"","questions":[{"question":"1+1=?","choices":["1","2"]}]}'
```

# Result

## Changed

- `jqv/prompt.py`: `prompt_hash(style)`（形式バージョン + PromptStyle の sha256 先頭 12 桁）と `PromptBuilder.hash`
- `jqv/model.py`: `default_style_for(model_id)` を切り出し
- `jqv/calibration.py`: `TemperatureScaler.meta`、`save(**meta)`（fitted_at 自動付与）、旧形式も読む `load`、`check_compatible`（`CalibrationMismatch` / `JQV_ALLOW_CALIBRATION_MISMATCH=1` で警告）、`info()`
- `jqv/types.py`: `CalibrationInfo`、`DecisionResponse.calibration` と `prompt_hash`
- `jqv/server.py`: モデルロード前に temperature ファイルの provenance を照合。`/health` と `/decision` に calibration を返す
- `scripts/eval.py`: JSON に `prompt_hash`, `n_val`, `choice_counts` を記録
- `scripts/fit_temperature.py`: companion JSON から provenance を埋めて保存（旧キャッシュは既定 style から hash を導出）
- `tests/test_calibration.py`: roundtrip、旧形式、model/prompt 不一致、prompt_hash、fit の回復
- `README.md`: provenance の節と API 例の注記

## Verified

- `uv run pytest`: 22 passed
- `fit_temperature.py` 再実行で mmlu / jmmlu の temperature JSON にメタデータが入った（T は 11.94 / 12.82 で不変）
- `load_calibration('Qwen/Qwen3-4B')` が `CalibrationMismatch` を送出（モデルロード前）、`JQV_ALLOW_CALIBRATION_MISMATCH=1` で警告のみ
- CPU でサーバ起動、`/health` と `/decision` に `calibration`（dataset=mmlu, n_val=400, prompt_hash 一致）が返る

## Deviations

- サーバ不一致の検証は uvicorn 起動ではなく `jqv.server.load_calibration` の直接呼び出しで行った（macOS に `timeout` コマンドがないため）。起動時の経路は同じ関数を通る

## Remaining

- なし
