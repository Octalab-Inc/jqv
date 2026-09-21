---
title: 正答確率を直接予測する calibrator（p_max / entropy / margin / legal_mass）で温度を超える domain 横断の校正を得る
status: draft
priority: P2
created_at: 2026-09-21T10:11:14+09:00
depends_on:
  - readout-codebook-legal-mass
---
# Goal

温度（scalar）では domain をまたぐと校正が崩れる（JevBench hard は部分転移、bridge は逆効果）問題に対し、
`p_max / entropy / probability_margin / legal_mass` などの決定ごとの信号から **正答確率 p_correct を直接予測する** 小さな calibrator を学習し、
out-of-domain（JevBench public hard、bridge）で温度より良い ECE と、精度 95% 条件での coverage を得る。`/decision` で `p_correct` として返す。

# Context

- 温度転移の結果（README「temperature の転移」節と JevBench 節）: MMLU ↔ JMMLU は転移、MMLU → JevBench hard は 0.274 → 0.107 の部分転移、bridge は逆効果。
- 選択的精度（README「精度の読み方」）: 32B で p ≥ 0.9 の 44% を精度 0.97 で自動処理できる。運用上は「自動処理してよいか」の判定が価値になる。
- 特徴量は `readout-codebook-legal-mass` の `scripts/eval.py --save-features` で npz に保存される。
- 学習に使ってよいもの: MMLU val 400、JMMLU val、（あれば）合成データの dev。JevBench public と bridge は test 専用で fit に使わない。
- NanoJev は ECE だけでなく NLL / Brier / risk-coverage / OOD を分けて見る方針、jevmlx は `calibrate` で temperature + multi calibrator を fit する。

# Scope

## In

- `jqv/calibrator.py`: 特徴量（p_max、entropy、margin、legal_mass、K、state / question の token 数）から p_correct を出す calibrator。
  候補はロジスティック回帰、p_max の isotonic、小さな GBM（sklearn）。データセット識別子のような漏洩特徴は使わない。
- `scripts/fit_calibrator.py`: fit（MMLU val + JMMLU val、任意で synth dev）→ 評価（MMLU test、JMMLU test、JevBench public hard、bridge）。
  指標: p_correct の ECE、AUROC、Brier、risk-coverage 曲線、精度 90% / 95% での coverage。比較対象は raw と温度（MMLU val で fit）。
- 保存形式は温度ファイルと同じ provenance（model、prompt_hash、fit データ、n）。`TemperatureScaler.check_compatible` 相当の互換チェック。
- サーバ: `JQV_CALIBRATOR_FILE` で読み込み、`/decision` に `p_correct` を追加（`calibrated_probabilities` は温度のまま）。
- README 節（温度との比較表、OOD での結論）。テスト。

## Out

- backbone の学習。
- タスク種別ごとの温度（baseline として 1 行載せるのみ）。
- 分布正解（probability fidelity）の校正（`proper-scoring-true-prob`）。

# Success

- 32B で、JevBench public hard と bridge の両方で calibrator の ECE が温度の ECE 以下、かつ精度 95% 条件の coverage が温度以上。
- MMLU test では温度より ECE が 0.01 以上悪化しない。
- `/decision` が `p_correct` を返し、互換チェックとテストがある。
- `results/calibrator_qwen3-32b.md` に raw / 温度 / calibrator の表（4 データセット × 指標）がある。

# Verify

```bash
uv run python scripts/fit_calibrator.py --model Qwen/Qwen3-32B --fit mmlu_val,jmmlu_val --eval mmlu_test,jmmlu_test,jevbench_hard,bridge   # 候補
uv run pytest tests/test_calibrator.py
JQV_CALIBRATOR_FILE=results/calibrator_qwen3-32b.json uv run uvicorn jqv.server:app --port 8000 & curl -s -X POST localhost:8000/decision -d '...'
```

# Open Questions

- モデルの種類（ロジスティック回帰で足りるか）。既定はロジスティック回帰 + isotonic の 2 つを比較。
- 合成データの dev を fit に含めるか（含めると OOD 評価の「OOD 度」が下がる）。既定は含めず、含めた場合を別行で報告。
