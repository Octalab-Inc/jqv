---
title: JevBench hard の弱い family（long_policy / temporal_numeric / probability）を狙った合成データを、正解をプログラム生成して作る
status: draft
priority: P1
created_at: 2026-09-21T10:11:14+09:00
depends_on: []
---
# Goal

JevBench hard で jqv-32B が弱い 3 family（long_policy 9/19、temporal_numeric 5/15、probability 4/10）を狙った合成データセットを、
**正解をプログラムで生成・検証した状態で** 用意する。family ごとに train ≥ 2,000 / dev 300 / test 500 問、JevBench 互換の JSONL で、
32B zero-shot の dev 精度が JevBench の当該 family の精度に近い（難易度が合っている）ことを確認済みにする。
あわせて JevBench 結果の family 別集計スクリプトを用意し、後続の学習タスクが同じ物差しで測れるようにする。

# Context

- JevBench public hard 111 問の構造（`/Users/h.imura/tmp/repo/jevbench/datasets/public/hard.jsonl`、キーは `state`, `question{type,instructions,criteria}`,
  `labels`, `expected`, `family`, `provenance.rationale`）:
  - long_policy 19 問: state 2,013〜3,746 token（中央値 3,016）。約款（定義、番号付き除外条項と例外、発効日付きの特約、サブリミット、免責）+ 請求ファイル + 表面的な答えへ誘導するメモ。K は 2〜6、type は choice 12 / noul 5 / score 2。
  - temporal_numeric 15 問: state 126〜695 token。月末規則、うるう年、時差、営業日、期限の包含規則など。state 内に誤った計算（例: 旧版の手順で計算した担当者のメモ）が置かれる。K 2〜6。
  - probability 10 問: state 268〜1,236 token。非復元抽出、条件付き確率、基準率、期待値。版が改訂された手順が distractor になる。K 2〜3、noul が多く、正解の確率そのものが計算できる。
- 32B zero-shot の family 別正解（README の JevBench 節）: adversarial 6/6、trap 8/8、routing 5/5、judge 14/17 に対し long_policy 9/19、multi_hop 10/18、temporal_numeric 5/15、probability 4/10、tradeoff 3/6。
- 既存の loader は `jqv/data.py`（`load_jsonl` は `data/bridge_synth.jsonl` 形式 `{"state","question","choices","answer"}`、`load_named`）。JevBench の question 形式（criteria dict）から choices への変換は `jqv/systemone.py:build_question` にある。
- 学習側 `jqv/train/data.py` は `cais/mmlu` しか読まない（JSONL 入力は後続タスクで追加）。
- 方針: 正解はプログラム（rule engine、`datetime`/`zoneinfo`、`fractions`）で生成し、LLM は表現の多様化だけに使う。JevBench public の問題文は test 専用で、生成データに一切流用しない。
- 生成の速度: Qwen3-32B の MPS 生成は遅い（単一ストリームで 10 tok/s 前後の見込み、未計測）。paraphrase を使うなら 14B のバッチ生成でスループットを先に測る。

# Scope

## In

- `jqv/synth/` に family ごとの generator を実装する:
  - `long_policy`: 定義 / 除外 / 例外 / 特約（発効日と適用条件）/ サブリミット / 免責 / 更新日を持つ約款をランダム生成し、事実関係（期間、占有状態、金額、日付）を持つ請求ケースと、表面的な答えに誘導するメモを付ける。決定ラベル（deny_〜 / pay_subject_to_〜 / pay_full_〜 など）は同じ規則を実行する rule engine が決める。state 2〜4k token、type は choice / noul / score を JevBench の比率で混ぜる。
  - `temporal_numeric`: 月末規則・うるう年・時差（`zoneinfo`）・営業日と祝日・期限の包含/排他・割合と単位の計算。正解は Python で計算し、state に誤計算の distractor を置く。
  - `probability`: 超幾何 / 二項 / 条件付き確率 / ベイズ / 期待値。`fractions` で厳密に計算し、`target_distribution`（noul なら p(yes)、choice なら各ラベルの真の確率）を保存する。後続の proper scoring タスクがこの真値を使う。
- 出力 `data/synth/<family>/{train,dev,test}.jsonl`（フィールド: `id`, `family`, `state`, `question{type,instructions,criteria}`, `labels`, `expected`, `target_distribution`（任意）, `rationale`（プログラムの計算過程）, `generator`, `seed`）。split は seed で分離し、同じ約款テンプレート・同じ数値の組が split をまたがないようにする。
- `jqv/data.py` に `load_named("synth:<family>:<split>")` を追加し、`scripts/eval.py` からそのまま評価できるようにする（choices への変換は `build_question` と同じ規則）。
- 表現の多様化: テンプレートの語彙・固有名・業種・文体を十分に振る。任意で 14B によるナラティブ部分の paraphrase（数値・日付・固有名が原文どおり残ることをチェッカーで検証し、失敗したら元に戻す）。paraphrase は先にスループットを測り、wall-clock 2 時間で打ち切る。
- 難易度確認: 14B と 32B の zero-shot（packed、MMLU val の T）で dev を評価し、JevBench の family 精度（32B: long_policy 47%、temporal_numeric 33%、probability 40%）と比較する。乖離が大きければ generator を調整して再測定する。誘導メモの「表面的な答え」をモデルがどれだけ選ぶか（surface-answer rate）も記録する。
- 汚染チェック `scripts/synth_contamination.py`: JevBench public 全 231 問との 8-gram 重複と固有名・ID の再利用を検出する。
- `scripts/jevbench_families.py`: `results/jevbench/<label>/hard/` の結果から family 別の正解数を出す（後続タスクの共通の物差し）。
- solver の単体テスト `tests/test_synth.py`（手計算済みのケースを含む）。

## Out

- 学習（`hard-family-targeted-lora`）。
- multi_hop / tradeoff の generator（必要なら別タスク）。
- JevBench の問題文やその paraphrase を学習データにすること。
- 英語以外のデータ。

# Success

- 3 family × train ≥ 2,000 / dev 300 / test 500 問が `data/synth/` にあり、全問の正解が solver 由来で、`tests/test_synth.py` が通る。
- `scripts/eval.py --dataset synth:<family>:dev` が動き、32B zero-shot の dev 精度が各 family で JevBench 精度 ±10 pt に入る（入らない場合は理由と調整履歴を `results/synth_difficulty.md` に残す）。14B は 32B より低い。
- 汚染チェックが通る（8-gram 重複のある問題が 1% 未満、固有名・ID の再利用なし）。
- `results/synth_difficulty.md` に model / T / 精度 / surface-answer rate / 生成と評価の所要時間が記録されている。
- 生成・評価は進捗行と ETA を出すログで実行されている。

# Verify

```bash
uv run pytest tests/test_synth.py
uv run python -m jqv.synth.generate --family long_policy --n-train 2000 --n-dev 300 --n-test 500 --seed 0   # 候補。実装後に確定
uv run python scripts/synth_contamination.py --jevbench /Users/h.imura/tmp/repo/jevbench/datasets/public
uv run python scripts/eval.py --dataset synth:long_policy:dev --model Qwen/Qwen3-32B --engine packed \
  --temperature-file results/mmlu_packed_qwen3-32b_temperature.json
uv run python scripts/jevbench_families.py    # 既存の 32B 結果で long_policy 9/19 などが再現されること
```

# Open Questions

- paraphrase の出所: ローカル 14B（遅いが無料）か、API の teacher（コスト）か、テンプレートのみか。既定はテンプレートのみ + 14B を 2 時間まで。
- multi_hop（10/18）も対象に加えるか。
