---
title: Qwen3-14B で語彙 readout (B) のベースラインと速度を 1.7B と同条件で測る
status: done
priority: P2
created_at: 2026-09-20T05:14:24+09:00
depends_on:
  - calibration-training
  - accuracy-fewshot-permavg
---

# Goal

Qwen3-1.7B と同じコード・プロンプト・データ・分割で Qwen3-14B の語彙 readout（B）を評価し、backbone 容量だけを変えたときの
精度・校正・スループットの変化を README のスケーリング表に 1 行追加する。14B を以後の開発モデル（LoRA・λ sweep・packed 実験）にする。

# Context

- ユーザーの方針: dense の 1.7B → 14B → 32B で backbone 効果を統制して見る。MoE（30B-A3B）と Qwen3.5（hybrid、Gated DeltaNet）は
  attention/KV 構造の議論がそのまま成り立たないため使わない。
- Qwen3-14B は BF16 で 29.5 GB（Apache-2.0）。M5 Max 128GB で packed S=8k・Q=1000 まで含めて余裕がある。ダウンロードは
  `results/download_scaling.log` で進行中（`hf download Qwen/Qwen3-14B`）。
- 全スクリプトは `--model` を受け取り、結果ファイルは `slug(model_id)`（`qwen3-14b`）で分かれる。temperature ファイルと head は
  model / prompt_hash を記録し不一致を拒否する。
- 1.7B の対応する値: MMLU 0.554 / ECE 0.413→0.080（T=12.0）、JMMLU 0.466 / 0.485→0.066、packed S=2038・Q=100 で 48.6 q/s、
  S=8038・Q=100 で 20.8 q/s。Jev（Hume 測定）: MMLU 0.918、ECE 0.031。
- 1.7B の MMLU 1,200 問評価は約 20 秒。14B は約 8 倍の演算なので数分。bench の naive は S=8k・Q=100 で 1.7B が 256 s なので
  14B では 30 分超。naive は S=2000 までに限定する。

# Scope

## In

- `JQV_MODEL=Qwen/Qwen3-14B` で `scripts/eval.py --engine packed`（MMLU 1200 / JMMLU 1200 / bridge 全件、seed 0）と
  `scripts/fit_temperature.py`、`scripts/transfer_temperature.py --model-slug qwen3-14b`
- `tests/test_engines_equivalence.py` 相当の確認を 14B でも 1 回実行（`JQV_MODEL` を指定して pytest。fp32 は 59 GB になるので bf16 で
  許容差 5e-2 の側で確認）
- `scripts/bench.py --model Qwen/Qwen3-14B`: S=2000 で generate/naive/kvcache/packed/shared × Q=10/100、S=8000 で packed/shared × Q=100/1000。
  結果は `results/bench_qwen3-14b.*`
- `scripts/isolation_test.py --model Qwen/Qwen3-14B`（secret-code のみで可）
- `scripts/scaling_table.py`（新規）: `results/*_qwen3-<size>*.json` から backbone 表
  （params, raw B acc, slot+LoRA acc, ECE raw / +T, packed q/s at S=2038 Q=100）を Markdown で出す。Jev 行は Hume の値を出典付きで固定
- README: 「Backbone スケーリング」節を新設し、表と 1.7B → 14B の差の読み（精度差は対応比較ではなく別モデルなので信頼区間で判断）を書く
- 学習・評価は必ず進捗ログ付きでバックグラウンド実行し、`results/*.jsonl` / JSON に逐次保存する

## Out

- LoRA 学習（`scaling-14b-lora-sweep`）
- 32B（`scaling-32b-final`）
- MLX への移植

# Success

- README のスケーリング表に Qwen3-14B の行（raw B の MMLU / JMMLU accuracy、ECE raw / +T、T、packed q/s）がある
- `results/mmlu_packed_qwen3-14b.json`, `results/jmmlu_packed_qwen3-14b.json`, `results/transfer_qwen3-14b.json`, `results/bench_qwen3-14b.md`,
  `results/isolation_qwen3-14b.json` が存在する
- 14B で naive / kvcache / packed / shared の choice logits が bf16 で 5e-2 以内に一致する
- 1.7B と比べた MMLU の差が 95% 信頼区間（各 ±3.4 ポイント）を超えるかどうかが README に書かれている

# Verify

```bash
JQV_MODEL=Qwen/Qwen3-14B uv run pytest tests/test_engines_equivalence.py tests/test_isolation.py
uv run scripts/eval.py --model Qwen/Qwen3-14B --dataset mmlu --engine packed --n 1200 --n-val 400
uv run scripts/eval.py --model Qwen/Qwen3-14B --dataset jmmlu --engine packed --n 1200 --n-val 400
uv run scripts/fit_temperature.py results/mmlu_packed_qwen3-14b.npz results/jmmlu_packed_qwen3-14b.npz
uv run scripts/bench.py --model Qwen/Qwen3-14B --engines generate naive kvcache packed shared --state-tokens 2000 --questions 10 100
uv run scripts/bench.py --model Qwen/Qwen3-14B --engines packed shared --state-tokens 8000 --questions 100 1000
uv run scripts/scaling_table.py
```

所要時間の目安: ダウンロード（29.5 GB）+ 評価 15 分 + bench 40 分。

# Result

## Changed

- `results/{mmlu,jmmlu,bridge}_packed_qwen3-14b*.{json,npz}`、`*_temperature.json`、`*_calibration.json`、`transfer_qwen3-14b.json`、`compare_{mmlu,jmmlu}_qwen3-14b.json`、`isolation_qwen3-14b.json`、`bench_qwen3-14b.{jsonl,json,md}`、`scaling_table.{md,json}`
- `scripts/scaling_table.py`: base model だけを行にし、perm_avg / 5-shot+perm_avg / slot+LoRA / q/s を列にした
- `tests/test_engines_equivalence.py`, `tests/test_isolation.py`: 許容差を dtype 依存に（bf16 は 1 ulp）
- `README.md`: 「Backbone スケーリング」節（表と 1.7B → 14B の読み）

## Verified

- `JQV_MODEL=Qwen/Qwen3-14B JQV_TEST_DTYPE=bfloat16 uv run pytest tests/test_engines_equivalence.py tests/test_isolation.py`: 16 passed（fp32 でも packed / shared / isolation を確認）
- eval（zero-shot / perm_avg / 5-shot / 5-shot+perm_avg）、fit_temperature、transfer_temperature、compare_runs、isolation_test、bench（S=2000 全 engine Q=10/100、S=8000 packed/shared Q=100/1000、perm_avg 1 条件）、scaling_table

## Deviations

- 5-shot（同 subject）と 5-shot + perm_avg を追加で測った（accuracy タスクの「同 subject 版のみ 1 回確認」に従う）
- compare_runs のシェル引数ミスで MMLU の対応比較を手動で再実行した

## Remaining

- なし
