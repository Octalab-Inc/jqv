# jqv — Qwen ベース Decision API（Jev 簡易版）

TypeSafe の Jev（[Hume の推定](https://archerhume.com/posts/jevs-architecture-unmasked/?v=3)）と
PFN の Preference API / [Insight Scan](https://www.preferred.jp/ja/news/pr20250508) に共通する
「**生成せず、共有 state に対する多数の質問へ確率分布を直接返す**」構造を、オープン LLM（Qwen3）だけで再現する PoC。

```
POST /decision
{"state": "橋梁Aでは主桁に腐食が確認された…",
 "questions": [{"question": "主桁の損傷は？", "choices": ["損傷なし","腐食","ひび割れ"]}, ...]}
→ {"decisions": [{"probabilities": [0.01, 0.97, 0.02], "calibrated_probabilities": [...], "confidence": 0.9}, ...]}
```

## 何を作ったか

同一モデル・同一プロンプトで、次の 4 つの推論構造を切り替えて比較できる。

| engine | 段階 | 構造 | 用途 |
|---|---|---|---|
| `generate` | A | 通常の `generate` で文字を出させて parse | ベースライン（生成する場合） |
| `naive` | B | 質問ごとに prefix+suffix をフル forward、末尾 logits の `A/B/C…` だけ読む | 「生成しない」だけの効果 |
| `kvcache` | D1 | state を 1 回 prefill → KV cache を複製して質問をバッチ | 共有計算（HF cache 方式） |
| `packed` | D2 | `[state \| q1 \| q2 \| …]` を 1 系列にし、block attention mask で各質問が state と自分だけを見る | Hume 推定の Jev 構造 |

`packed` の attention mask と position id:

```
            state     q1    q2    q3
state       ◤causal
q1          █████     ◤
q2          █████           ◤
q3          █████                 ◤        position_ids: state 0..S-1, 各 q は S から再開
```

fp32 では `naive` / `kvcache` / `packed` の choice logits が 1e-4 以内で一致する（`tests/test_engines_equivalence.py`）。
つまり D2 は「Q 個の独立した prefix+q_i forward」と数値的に同じ計算を、prefix の K/V を 1 部だけ持って 1 回の forward で行う。

readout は語彙 logits のうち選択肢文字 `" A"`, `" B"`, … の 1 token だけを softmax する（ラベル名を直接読まない）。
`confidence` は Hume の言う post-hoc 指標（一様分布からの距離 `1 - H(p)/log K`）で、確率ではない。

## セットアップ

```bash
uv sync                        # Python 3.12, torch (MPS), transformers 5.x
uv run pytest                  # 初回に Qwen/Qwen3-1.7B をダウンロード
```

macOS / Apple Silicon（MPS）を前提に HF Transformers だけで動く。CUDA 環境でもそのまま動く。
モデルは `JQV_MODEL`（既定 `Qwen/Qwen3-1.7B`）、dtype は `JQV_DTYPE`（既定 bf16）で変更。
`*-Base` モデルを指定すると chat template を使わない plain プロンプトに自動で切り替わる。

## 使い方

```bash
# API サーバ
uv run python -m jqv.server --engine packed --temperature-file results/mmlu_packed_qwen3-1.7b_temperature.json
curl -s localhost:8000/decision -H 'content-type: application/json' -d @- <<'JSON'
{"state": "橋梁A: 主桁下フランジに広範囲の腐食。床版にひび割れなし。支承は良好。",
 "questions": [{"question": "主桁の損傷は何か。", "choices": ["損傷なし","腐食","ひび割れ"]},
               {"question": "床版にひび割れはあるか。", "choices": ["ある","ない"]}]}
JSON

# Python
from jqv.model import load_runtime
from jqv.engine import make_engine
from jqv.types import Question
rt = load_runtime()
eng = make_engine("packed", rt, temperature=12.9)
eng.decide(state, [Question(question="…", choices=["…", "…"])])
```

## 実験スクリプト

```bash
uv run scripts/eval.py --dataset mmlu  --engine packed --n 1200 --n-val 400   # accuracy / NLL / Brier / ECE
uv run scripts/eval.py --dataset jmmlu --engine packed --n 1200 --n-val 400
uv run scripts/fit_temperature.py results/mmlu_packed_qwen3-1.7b.npz          # T を val で学習、test で前後比較 + reliability diagram
uv run scripts/bench.py --state-tokens 500 2000 8000 --questions 1 10 100     # engine 別の latency / questions/s
uv run scripts/isolation_test.py                                              # Hume の secret-code 実験と 5 番目選択肢実験
```

- `eval.py` は同じ state を持つ質問をまとめて 1 回の `decide()` に渡す（MMLU/JMMLU は state 空で N 問を一括）。
- `fit_temperature.py` は val 分割で temperature scaling を学習し、test 分割で ECE / Brier / NLL の前後を出す。
- `isolation_test.py` は `packed_causal`（block mask を使わない素朴な連結）を負対照として、兄弟質問の情報が漏れないことを示す。
- `bench.py` は各条件の完了ごとに結果を `results/bench_<model>.jsonl` へ追記し、残り時間の推定を表示する。中断後に同じコマンドで再開できる。

結果は `results/` に JSON / PNG / Markdown で出る。

## 結果（Qwen3-1.7B, bf16, M5 Max）

### 精度と校正（`packed` engine, val 400 / test 800, T は val で学習）

| dataset | accuracy | ECE before | ECE after | Brier before | Brier after | NLL before | NLL after | T |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| MMLU (en)  | 0.554 | 0.413 | **0.077** | 0.852 | 0.576 | 5.79 | 1.08 | 11.9 |
| JMMLU (ja) | 0.466 | 0.485 | **0.064** | 0.983 | 0.639 | 6.13 | 1.19 | 12.8 |
| bridge_synth (30問, 校正なし) | 0.933 | 0.051 | - | 0.079 | - | 0.10 | - | - |

生の 1-token readout は平均 confidence 0.96 と極端に過信しているが、スカラー 1 個の temperature で ECE は 0.06〜0.08 まで下がる
（Hume が Jev で測った ECE 0.031 にはまだ届かない）。reliability diagram は `results/*_reliability_{before,after}.png`。

### Isolation（Hume の secret-code 実験の再現）

| engine | p(秘密) 単独 | 兄弟質問に秘密 | state に秘密 |
|---|---:|---:|---:|
| naive / kvcache / packed | 0.000 | **0.000** | 1.000 |
| packed_causal（負対照: 素朴な連結） | 0.000 | **0.996** | 1.000 |

block mask を入れた `packed` は兄弟質問の情報を一切見ない（Jev と同じ挙動）。mask を外すと完全に漏れる。

5 番目の無関係選択肢を足したときの正解 vs 最有力誤答の log-odds 変化は −0.41 ± 0.41（n=21, bridge_synth）で、
Hume が Jev で観測した −0.28 と同じ向き。語彙 readout でも選択肢列がプロンプト内で相互作用するので、独立 logit + softmax にはならない。

### スループット

Qwen3-1.7B / bf16 / Apple M5 Max。各条件 warmup 1 回 + 3 回計測の中央値（warmup が 60 秒を超えた条件は 1 回）。
質問は 1 問あたり約 55 token。全 36 条件は `results/bench_qwen3-1.7b.md` にある。

| state tok | Q | generate (A) | naive (B) | kvcache (D1) | packed (D2) | D2 speedup vs B |
|---:|---:|---:|---:|---:|---:|---:|
| 538  | 10  | 0.61 s | 0.53 s | 0.14 s | **0.12 s** | 4.5x |
| 538  | 100 | 7.2 s  | 6.8 s  | **1.08 s** | 1.20 s | 5.7x |
| 2038 | 10  | 3.1 s  | 3.3 s  | 0.44 s | **0.40 s** | 8.2x |
| 2038 | 100 | 43 s   | 41 s   | 2.3 s  | **2.1 s**  | 20x |
| 8038 | 10  | 23 s   | 23 s   | **2.1 s** | 2.4 s | 9.8x |
| 8038 | 100 | 231 s  | 256 s  | 6.6 s  | **4.8 s**  | **53x** |

Q=1 では 4 engine とも同等（共有するものがない。S=8038 で naive / packed とも 0.79 s）。

読み取れること:

1. **「生成しない」だけでは速くならない。** generate と naive は 0.9〜1.1x で同じ。prefill が支配的で、4 token のデコードは誤差。
   B の価値は速度ではなく「確率分布が出て校正できる」ことにある。
2. **共有計算の効果は state 長 × 質問数に比例して伸びる。** S=8k・Q=100 で packed は naive の 53 倍、kvcache は 39 倍。
   Hume の「state 長が支配し、質問の追加コストはほぼゼロ」という Jev の観測と同じ形になる。
3. **packed は state が長いほど kvcache より有利。** kvcache は HF の cache を batch 行ごとに複製するため 8k state ではメモリ制約で batch が 8 に落ちる。
   packed は prefix K/V を 1 部しか持たない。短い state では mask 構築のオーバーヘッドで kvcache が僅かに速い。
4. 絶対値は Jev（30k token を約 160 ms）より 1〜2 桁遅い。これは 1.7B を MPS で動かしている環境差で、構造の比較には影響しない。

## 設計メモ

- **prefix / suffix の分割 tokenize**: state 側と質問側を別々に tokenize し、全 engine が同じ token id 列を使う。
  境界で BPE の merge が起きないことをテストで確認している（`tests/test_prompt.py`）。
- **`kvcache` のメモリ**: HF の `DynamicCache` は batch 行ごとに prefix の K/V を実体化するため、
  長い state では batch を `cache_budget_bytes` で自動的に絞る。`packed` は prefix K/V を 1 部しか持たないのでこの問題がない。
  vLLM の prefix caching（paged KV）は前者を共有化する仕組みで、Linux/CUDA へ持っていくならそれで `kvcache` を置き換えられる。
- **長い packed 列**: `PackedEngine(max_tokens=…)` を超える場合は prefix を cache に入れ、質問側だけを chunk で pack する（同じ mask、query 行 = 質問のみ）。
- **明示 mask のコスト**: 4D mask を渡すと sdpa の causal 高速パスが使えず、8k 系列で約 2 倍遅くなる。
  そのため分岐が 1 本のとき（packed）と padding が不要なとき（naive）は mask を渡さない。
- **bench.py は条件ごとに JSONL へ追記し、再実行時は完了済み条件をスキップする。** 途中で止めても失うのは実行中の 1 条件だけ。
- **校正**: 1-token readout の生 logits は非常に尖っており（MMLU で mean confidence ≈ 0.96, ECE ≈ 0.38）、
  temperature scaling だけで ECE は 0.06 前後まで落ちる。「0.8 と言ったら 8 割当たる」の最初の一歩はモデル改造なしで到達できる。

## 次フェーズ（未実装）

- **C: 専用 readout head** — `engine/head.py` で最終 hidden state → `nn.Linear(hidden_size, max_choices)`。LLM は freeze。
  語彙 15 万次元ではなく選択肢次元を直接出す。Hume の「slot head」相当。
- **E: 校正指向の post-training** — `L = CE + λ·Brier` で head または LoRA を学習（TRL/PEFT）。その先で RL / preference optimization。
- **本番サービング** — Linux GPU で vLLM + Automatic Prefix Caching、同じ `/decision` エンドポイント。
