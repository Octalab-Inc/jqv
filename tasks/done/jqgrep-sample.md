---
title: jqv を使った cascade 型 semantic code search のサンプル jqgrep を作る（字句選別 → sketch 判定 → 全文 shared-state で range 判定）
status: done
priority: P2
created_at: 2026-09-21T17:18:04+09:00
depends_on: []
---

# Goal

jqv を「多数の decision を本当に 1 つの state に掛けるアプリ」として使うサンプル `jqgrep` を作る。自然言語の検索意図を受け取り、
埋め込みも index も持たずにファイルツリーを live に探索し、`path:start-end` の line range を確率付きで返す。
Stage 2 では 1 ファイル全文を state にして、複数の line range の関連判定と passage の役割分類（実装 / 呼び出し / テスト / 設定）を
1 回の packed / shared forward で出す。

# Context

- 参考は jegrep（Jev で grep する Rust 製ツール）の `cascade` 戦略: sketch で候補を routing → full-source passage を検証。line range を返し、
  1 ファイル最大 16 range。ただし jegrep の「absolute yes/no probability なので閾値が意味を持つ」という主張は jqv ではまだ言えない
  （温度は MMLU ↔ JMMLU では転移するが bridge では逆効果。README の温度転移節）。
- jqv 側の材料: `jqv.model.load_runtime`、`jqv.engine.make_engine("packed"|"shared", rt, temperature=...)`、`eng.decide(state, [Question(...)])`
  が `Decision.probabilities` / `calibrated_probabilities` を返す。packed は `chunk_tokens=2048` で長い質問列を分割する。
  サーバ `POST /decision` も同じ形。温度ファイルは `results/mmlu_packed_<model>_temperature.json`。
- 速度の性質（README スループット節）: state が長く質問が多いほど共有計算が効く（S=8k・Q=100 で naive の 53〜73 倍）。
  query を state にしてファイルを質問にする構成は state が短いので利得が小さい。ファイル全文を state にして range を質問にする構成が本命。
- 既定 backbone は 14B（速度と精度のバランス）、精度優先で 32B。開発・テストは 1.7B で行う（学習ジョブと GPU を共有するため小さく）。

# Scope

## In

- `jqgrep/` パッケージ（`walker.py`, `sketch.py`, `candidate.py`, `passages.py`, `search.py`, `output.py`, `jqv_client.py`, `__main__.py`）。
- Stage 0（LLM なし）: `git ls-files` または os.walk で探索、`.gitignore` 相当と binary / 巨大ファイルを除外、path・拡張子・top-level symbol・
  import・query 語の lexical hit（行と前後）・ファイルサイズから候補 100〜200 に絞る。
- Stage 1（cheap semantic routing）: 各候補の sketch（path、言語、symbol、先頭 384〜1,024 bytes、grep 周辺）を質問にして yes/no を jqv で判定し、
  上位 10〜30 ファイルに絞る。
- Stage 2（full semantic verification）: 上位ファイルの全文（上限 token で window 分割）を state にし、line range（既定 40 行・stride 30、
  1 window 最大 32 range）ごとの関連判定に加えて、ファイル全体の関連、役割（primary implementation / call site / test or example /
  config or docs）、曖昧性を同じ `decide` で問う。隣接・重複 range を merge。
- ランキング主体の選別: top-k、`p > max_p − gap` の相対閾値、最終段だけ厳しめ。絶対閾値 `--threshold` はオプションとして残すが既定では使わない。
- 出力: テキスト（score、`path:start-end`、役割、snippet）と `--json`。
- CLI: `python -m jqgrep "query" path [--model Qwen/Qwen3-14B] [--engine packed|shared] [--server http://localhost:8000] [--json] [--top-files N] [--k N]`。
  `--server` は起動済みの jqv サーバ（`/decision`）を使う。
- `jqgrep/README.md`（英語、使い方と設計、限界）、top README への 1 段落、テスト（walker / sketch / range merge / 相対閾値。モデル不要）。
- 動作確認: jqv リポジトリ自身を対象に 3 クエリを 1.7B で実行し、期待ファイル（例: block mask → `jqv/engine/packed.py`）が上位に来ることを記録。

## Out

- 検索 benchmark（ripgrep / BM25 / embedding / naive / packed / shared / Jev の比較）とラベル付きデータセット、コード検索用の温度校正。
  → 別タスク候補。
- jegrep の他 strategy（beam / sniff / window）の移植。
- index や埋め込みのキャッシュ。

# Success

- `python -m jqgrep "where is the block attention mask for packed sequences built" .` が jqv リポジトリで `jqv/engine/packed.py` の該当 range を上位 3 に返す
  （1.7B、既定設定）。他 2 クエリ（温度の学習、TypeSafe wire format）も期待ファイルが上位 3 に入る。結果を `jqgrep/README.md` に記録。
- Stage 2 が 1 ファイルあたり 1 回の `decide`（window 分割時は window ごと 1 回）で range + 役割 + 曖昧性を返す。
- モデル不要のテストが通る（`tests/test_jqgrep.py`）。
- `--json` 出力と `--server` 経由の実行が動く。

# Verify

```bash
uv run pytest tests/test_jqgrep.py
uv run python -m jqgrep "where is the block attention mask for packed sequences built" . --model Qwen/Qwen3-1.7B
uv run python -m jqgrep "fitting the calibration temperature" . --model Qwen/Qwen3-1.7B --json | head -40
JQV_MODEL=Qwen/Qwen3-1.7B uv run uvicorn jqv.server:app --port 8012 & uv run python -m jqgrep "TypeSafe wire format" . --server http://localhost:8012
```

# Open Questions

- なし（設計はユーザー提示の 3 段 cascade に従う）。

# Result

## Changed

- `jqgrep/` package: `walker.py` (git ls-files / os.walk, binary and size filters), `sketch.py` (query terms, symbols, imports,
  lexical score with a code-over-docs/data prior, `--include-docs`), `candidate.py` (stage 1: groups of 12 sketches in one shared
  state, per-file yes/no + listwise "which file", final pass over the best 2k), `passages.py` (line ranges, windows, merge, trim,
  role questions), `search.py` (stage 2: one `decide` per file window with range yes/no + listwise range + file relevance / role /
  production / ambiguity; relative-gap ranking), `output.py` (text / JSON), `jqv_client.py` (in-process runtime or `--server`),
  `__main__.py` (CLI). `jqgrep/README.md` (design, options, results, limits); a paragraph in the top-level README.
- `tests/test_jqgrep.py`: 5 model-free tests (walker, sketch, ranges, merge/trim, the whole cascade with a fake client).

## Verified

- `uv run pytest tests/test_jqgrep.py`: 5 passed.
- Qwen3-14B, default settings: all three reference queries return the expected file as the sole top hit
  (`jqv/engine/packed.py:31-69` 0.91, `scripts/fit_temperature.py:31-68` 0.90, `jqv/systemone.py:17-77` 0.77), 145-173 s per query
  including model load (`results/jqgrep_14b.log`, kept outside git).
- Qwen3-1.7B, default settings: query 1 expected file #3, query 2 #2, query 3 not in the top 3.
- `--json` and `--server` (against a temporary 1.7B jqv server on port 8012) run end to end.
- Stage 2 uses one `decide` per file window carrying range + role + ambiguity questions.

## Deviations

- The Success criterion asked for top-3 on all three queries with the 1.7B; the third query (TypeSafe wire format) fails
  there because the 1.7B says yes to nearly every sketch and task notes quoting the code outrank the code. Recorded in the
  README; the recommended backbone is 14B, where all three are #1.
- Stage 1 was changed from independent per-sketch questions (the task text) to grouped shared states with a listwise question,
  because the independent version let the 1.7B accept everything.
- A slip while closing the task briefly committed the jqgrep README over the top-level README (854d234); fixed in the next commit.

## Remaining

- No labelled code-search benchmark (ripgrep / BM25 / embeddings / jqv engines / Jev) and no calibration for `--threshold`;
  scores are comparable within a query only.
- `--gap 0.25` leaves one file on single-implementation queries with the 14B; multi-file intents need `--gap 0.5`.
- Stage 0 falls back to the smallest files when a query has no lexical overlap.
