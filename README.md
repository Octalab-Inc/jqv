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
| `readout="rows"` | B' | 全語彙 projection を行わず、LM head の選択肢文字の行だけで logits を計算（全 engine で選択可） | 「vocab projection を捨てても同じ」の証明 |
| `kvcache` | D1 | state を 1 回 prefill → KV cache を複製して質問をバッチ | 共有計算（HF cache 方式） |
| `packed` | D2 | `[state \| q1 \| q2 \| …]` を 1 系列にし、block attention mask で各質問が state と自分だけを見る | Hume の観測と整合する block/tree attention 実装の一つ（reference 実装） |
| `shared` | D3 | packed と同じ入力を、mask を使わないカスタム attention で計算。branch の query をまとめて shared prefix に attention（Hydragen 型分解）し、branch 内は小さな causal attention、log-sum-exp で合成 | block sparsity を演算量・メモリで実際に使う実装（L×L の mask を持たない） |

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
`readout="rows"` (B') は同じ logits を `h @ W[choice_ids].T` で直接計算する。fp32 で B と 1e-4 以内で一致し
（`tests/test_engines_equivalence.py`）、15 万次元の projection は決定に不要であることを示す。
`confidence` は Hume が TypeSafe 公式 adapter で確認した post-hoc 指標 `(p_max − 1/K) / (1 − 1/K)` で、確率ではない
（一様分布で 0、one-hot で 1）。entropy 版 `1 − H(p)/log K` は `entropy_concentration` として別に返す。

**Qwen3 の thinking は無効化して固定している。** Qwen3 の chat template は既定で `enable_thinking=True` なので、
そのまま assistant 直後の logits を読むと「本当は `<think>` を始めたい位置で無理やり A/B/C を比較する」ことになる。
jqv は assistant 側を `<think>\n\n</think>\n\n` + `Answer:` で始める非 thinking プロンプトを `jqv/prompt.py` に固定し、
`tests/test_prompt.py` が公式 `apply_chat_template(enable_thinking=False)` と完全一致することを検証している。
generate (A) も同じ prefix + suffix を使うので、A/B/D1/D2 の比較に reasoning 有無の差は混じらない。

**packed は Jev の再現ではなく、Jev で観測された挙動（shared state、sibling isolation、確率の直接 readout）を
open な Qwen で再現できることを示す reference 実装である。** Hume 自身も sibling isolation は tree mask 以外の仕組みでも
実現でき、exact な attention mask までは分からないと留保している。

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
# API サーバ（temperature ファイルは MMLU-en で学習した T=11.9 の流用。橋梁点検ドメインでは未検証。
# 応答の calibration.dataset にその出所が入る）
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
  T は [0.05, 100] の対数格子 + 黄金分割で NLL 最小化する（少数・ほぼ分離可能なデータで発散しないため）。
- `transfer_temperature.py` は npz キャッシュだけを使い、source × target の全組合せで T を転移させた ECE / NLL / Brier を出す。
- `isolation_test.py` は `packed_causal`（block mask を使わない素朴な連結）を負対照として、兄弟質問の情報が漏れないことを示す。
- `permutation_test.py` は `--mode label`（文字だけ巡回）、`--mode order`（並びだけ巡回）、`--mode fifth`（無関係な 5 番目を追加）で
  確率の変動を測る。`Question.labels` で位置ごとの文字を指定できる（実験用）。
- `bench.py` は各条件の完了ごとに結果を `results/bench_<model>.jsonl` へ追記し、残り時間の推定を表示する。中断後に同じコマンドで再開できる。

結果は `results/` に JSON / PNG / Markdown で出る。

### 校正温度の出所（provenance）

温度は学習した分布にしか通用しない。`fit_temperature.py` が書く temperature ファイルには
`model`, `engine`, `prompt_hash`（プロンプト形式と system prompt から算出）, `dataset`, `n_val`, `choice_counts`, `fitted_at` が入り、
サーバは起動時に `model` と `prompt_hash` を実行時と照合する。不一致なら（モデルをロードする前に）起動に失敗し、
`JQV_ALLOW_CALIBRATION_MISMATCH=1` のときだけ警告で続行する。旧形式（`temperature` だけ）のファイルも読めるが未検証扱いになる。
`/decision` の応答には `calibration`（temperature と上記メタデータ）が入り、temperature 未設定なら `null`。
`/health` にも同じ情報と `prompt_hash` が出る。

## 結果（Qwen3-1.7B, bf16, M5 Max）

### 精度と校正（`packed` engine, val 400 / test 800, T は val で学習）

| dataset | accuracy | ECE before | ECE after | Brier before | Brier after | NLL before | NLL after | T |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| MMLU (en)  | 0.554 | 0.413 | **0.080** | 0.852 | 0.576 | 5.79 | 1.08 | 12.0 |
| JMMLU (ja) | 0.466 | 0.485 | **0.066** | 0.983 | 0.639 | 6.13 | 1.19 | 12.8 |
| bridge_synth (30問, 校正なし) | 0.933 | 0.051 | - | 0.079 | - | 0.10 | - | - |

生の 1-token readout は平均 confidence 0.96 と極端に過信しているが、スカラー 1 個の temperature で ECE は 0.06〜0.08 まで下がる
（Hume が Jev で測った ECE 0.031 にはまだ届かない）。reliability diagram は `results/*_reliability_{before,after}.png`。

### temperature の転移（`scripts/transfer_temperature.py`, `results/transfer_qwen3-1.7b.json`）

source の val で学習した T を、別の target の test にそのまま適用した。Jev の ECE 0.031 は zero-shot 値なので、
in-distribution の対角ではなく非対角がそれとの比較対象になる。

| T の学習元 → 適用先 | T | MMLU test (n=800) ECE | JMMLU test (n=800) ECE | bridge (n=30) ECE |
|---|---:|---:|---:|---:|
| なし (T=1) | 1.00 | 0.413 | 0.485 | **0.051** |
| MMLU val | 11.96 | **0.080** | 0.082 | 0.237 |
| JMMLU val | 12.84 | 0.081 | **0.066** | 0.266 |
| MMLU+JMMLU val | 12.34 | 0.085 | 0.072 | 0.253 |
| oracle（各 target の test 自身で学習） | 11.5 / 12.9 / 2.94 | 0.092 | 0.070 | 0.047 |

- **言語をまたいだ転移は成立する。** MMLU で学習した T=12.0 を JMMLU に当てても ECE 0.082（in-distribution 0.066）、逆向きも 0.081（0.080）。
  英語と日本語の知識問題で、生 logit の scale の歪みはほぼ同じ（T ≈ 12）。
- **タスク種別をまたいだ転移は成立しない。** bridge（state に答えが書いてある読解型、正答率 0.93）では生の確率が既にほぼ校正されており
  （ECE 0.051、oracle T=2.9）、T=12 を当てると平均 confidence が 0.98 → 0.73 に落ちて過小確信になり ECE は 0.24 に悪化する。
- つまり Qwen3-1.7B の 1-token logit の歪みは「一定の scale」ではなく、closed-book の知識問題では約 12 倍、state から読み取れる問題では約 3 倍と
  タスクの性質で変わる。post-hoc の scalar 1 個では汎用 Decision API の校正にはならず、ここが Jev（RLCD で学習した分布）との差になる。
  bridge は 30 問の合成データなので数値は目安。転移の判定基準は「非対角 ECE が対角 + 0.02 以内」とした。

### 文字ラベルの prior（`scripts/permutation_test.py --mode label`, `results/permutation_label_qwen3-1.7b.json`）

選択肢の位置と内容を固定したまま、前に付ける文字だけを巡回シフトする（`A. x / B. y / C. z` → `B. x / C. y / A. z` → …）。
意味的選択肢に戻して集計するので、残る差は「どの文字 token を読むか」だけ。生の確率（T=1）で測定。

| dataset | 文字ごとの平均確率 A / B / C / D（一様なら 0.25） | p(correct) のシフト間平均絶対差 | argmax が全シフトで一致 | accuracy の範囲 |
|---|---|---:|---:|---|
| MMLU (n=300) | 0.31 / 0.28 / 0.22 / 0.19 | 0.174 | 54% | 0.533〜0.560 |
| bridge (n=30) | 0.30 / 0.31 / 0.24 / 0.24 | 0.039 | 93% | 0.900〜0.955 |

- **文字 token の prior は無視できない。** MMLU では A/B が C/D より系統的に高く（0.31 vs 0.19）、文字を付け替えるだけで 46% の問題で
  argmax が変わる。accuracy の平均はほぼ変わらない（0.53〜0.56）ので、prior は「迷っている問題」を A/B 側に倒している。
- state に答えが書いてある bridge では効果が小さい（一致 93%）。証拠が強いと prior は上書きされる。
- 注意: 文字を巡回させると `B. C. D. A.` のように文字が非順序で並ぶ不自然なプロンプトになる。この設計は「文字の効果」だけを取り出す
  代わりに、その不自然さも含んで測っている。並び順そのものの効果は次の order 実験で測る。

### 選択肢の並び順と 5 番目選択肢（`--mode order`, `--mode fifth`）

order: 文字は A, B, C, D の順に固定し、選択肢テキストの並びだけを巡回シフトする（listwise の位置効果）。label 実験と同じ指標。

| dataset / mode | slot ごとの平均確率（label: 文字 A/B/C/D、order: 位置 1/2/3/4） | p(correct) の平均絶対差 | argmax 一致 | accuracy の範囲 |
|---|---|---:|---:|---|
| MMLU / label (n=300) | 0.31 / 0.28 / 0.22 / 0.19 | 0.174 | 54% | 0.533〜0.560 |
| MMLU / order (n=300) | 0.25 / 0.31 / 0.22 / 0.22 | **0.220** | **48%** | 0.547〜0.580 |
| bridge / label (n=30) | 0.30 / 0.31 / 0.24 / 0.24 | 0.039 | 93% | 0.900〜0.955 |
| bridge / order (n=30) | 0.29 / 0.32 / 0.24 / 0.23 | 0.084 | 87% | 0.909〜0.967 |

- **並び順の効果は文字 prior より大きい。** MMLU で並びを変えると 52% の問題で argmax が変わり、p(correct) は平均 0.22 動く。
  位置 2 が最も選ばれやすい（0.31）。文字 prior（A/B 寄り）と位置 prior（2 番目寄り）は別物として存在する。
- 両実験とも accuracy の平均はほとんど動かないので、これらの prior は「証拠が弱い問題の tie-break」として働いている。
  校正前の確率が one-hot に近いため、tie-break の向きが変わるだけで p(correct) が 0↔1 で入れ替わり、平均絶対差が大きく出る。
- Hume は Jev でも option order で確率が動くと報告している。語彙 readout の Qwen ではその感度がかなり大きい。順序を平均する
  （全巡回シフトの平均を返す）だけで実用上の対策になるが、コストは K 倍になる。pointer head の学習で順序 shuffle を入れるのが本筋。

fifth: 4 択に無関係な 5 番目「該当なし／不明」を追加し、正解 vs 最有力誤答の log-odds の変化を測る（Hume の Jev: −0.28、95% CI −0.36〜−0.19）。

| dataset | n | Δlog-odds 平均 ± 95% CI | 個別の SD | 5 番目に乗る平均確率 | 上位 4 択内で argmax が変わる割合 |
|---|---:|---:|---:|---:|---:|
| MMLU | 300 | −0.04 ± 0.45 | 3.96 | 0.19 | 6.7% |
| bridge | 21 | −0.49 ± 0.39 | 0.92 | ≈0 | 0% |

- MMLU では **系統的なずれは検出されない**（CI が 0 をまたぐ）が、個別には ±4 nat も動き（SD 3.96）、6.7% で 4 択内の argmax が変わる。
  つまり選択肢リストは final hidden state に joint に影響しており（独立 logit + softmax ではない）、ただし Jev のように一定方向に
  縮む形ではなく、問題ごとに大きく揺れる。
- bridge では Jev と同じ向き（−0.49）だが n=21 で CI が 0 をかろうじて外れる程度。
- MMLU で 5 番目に 19% も確率が乗るのは、英語の知識問題に日本語の「該当なし／不明」を足したためでもあり、選択肢文言の言語を揃えた再測定が必要。

### Isolation（Hume の secret-code 実験の再現）

| engine | p(秘密) 単独 | 兄弟質問に秘密 | state に秘密 |
|---|---:|---:|---:|
| naive / kvcache / packed | 0.000 | **0.000** | 1.000 |
| packed_causal（負対照: 素朴な連結） | 0.000 | **0.996** | 1.000 |

block mask を入れた `packed` は兄弟質問の情報を一切見ない（Jev と同じ挙動）。mask を外すと完全に漏れる。

5 番目選択肢の実験は上の「選択肢の並び順と 5 番目選択肢」節（n=300）を参照。`isolation_test.py` 内の n=21 版は古い測定。

### スループット

Qwen3-1.7B / bf16 / Apple M5 Max。各条件 warmup 1 回 + 3 回計測の中央値（warmup が 60 秒を超えた条件は 1 回）。
質問は 1 問あたり約 55 token。全 36 条件は `results/bench_qwen3-1.7b.md` にある。

| state tok | Q | generate (A) | naive (B) | kvcache (D1) | packed (D2) | shared (D3) | 最良 vs B |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 538  | 10  | 0.61 s | 0.53 s | 0.14 s | **0.12 s** | 0.13 s | 4.5x |
| 538  | 100 | 7.2 s  | 6.8 s  | 1.08 s | 1.20 s | **0.95 s** | 7.2x |
| 2038 | 10  | 3.1 s  | 3.3 s  | 0.44 s | **0.40 s** | 0.43 s | 8.2x |
| 2038 | 100 | 43 s   | 41 s   | 2.3 s  | 2.1 s  | **1.8 s** | 23x |
| 2038 | 1000 | -     | -      | -      | **9.9 s** | 13.3 s | - |
| 8038 | 10  | 23 s   | 23 s   | 2.1 s | 2.4 s | **1.6 s** | 14x |
| 8038 | 100 | 231 s  | 256 s  | 6.6 s  | 4.8 s  | **3.5 s** | **73x** |
| 8038 | 1000 | -     | -      | 64 s   | **15.9 s** | 22.5 s | - |

Q=1000 は naive / generate では 40 分以上かかるため未計測。packed の Q=1000 は prefix を KV cache に入れ、質問側を 2048 token ずつ
pack する chunk 動作（`chunk_tokens=2048`。16384 では 36 s で、小さいタイルほど masked された branch-branch ブロックの無駄が減る）。

Q=1 では 4 engine とも同等（共有するものがない。S=8038 で naive / packed とも 0.79 s）。

B vs B'（readout full / rows、packed、Q=100、同一プロセスで交互に 5 回計測した中央値、`results/readout_ab_qwen3-1.7b.json`）:

| state tok | full (B) | rows (B') | rows / full |
|---:|---:|---:|---:|
| 538  | 1.19 s | 1.14 s | 0.96 |
| 2038 | 1.62 s | 1.56 s | 0.96 |

差は約 4%（Q=100 で 40〜60 ms、LM head の (100 × 151k) projection と softmax の分）で、backbone に比べて無視できる。
B' の価値は速度ではなく「語彙 projection なしでも決定が同一」という切り分けにある。
なお `results/bench_qwen3-1.7b.md` の `:rows` 行は別プロセスで測ったもので、run 間のばらつき（±30%）を含む。

読み取れること:

1. **「生成しない」だけでは速くならない。** generate と naive は 0.9〜1.1x で同じ。prefill が支配的で、4 token のデコードは誤差。
   B の価値は速度ではなく「確率分布が出て校正できる」ことにある。
2. **共有計算の効果は state 長 × 質問数に比例して伸びる。** S=8k・Q=100 で packed は naive の 53 倍、kvcache は 39 倍。
   Hume の「state 長が支配し、質問の追加コストはほぼゼロ」という Jev の観測と同じ形になる。
3. **packed は state が長いほど kvcache より有利。** kvcache は HF の cache を batch 行ごとに複製するため 8k state ではメモリ制約で batch が 8 に落ちる。
   packed は prefix K/V を 1 部しか持たない。短い state では mask 構築のオーバーヘッドで kvcache が僅かに速い。
5. **shared (D3) は 1 回の forward に収まる範囲（prefix + 全質問 ≤ 16k token）で packed より 1.15〜1.5 倍速く、Q=1000 の chunk 領域では
   packed の小タイルに負ける（下の「D3」節）。**
4. 絶対値は Jev（30k token を約 160 ms）より 1〜2 桁遅い。これは 1.7B を MPS で動かしている環境差で、構造の比較には影響しない。

## D3: mask を持たない shared-prefix attention（`jqv/engine/shared.py`）

packed (D2) は `(1, 1, L, L)` の mask を SDPA に渡すため、attention は dense に計算される。D3 は同じ packed 入力
（`[prefix | q1 | … | qQ]`、branch ごとに prefix 直後から再開する position id）を、Transformers 5.x の
`AttentionInterface.register` で差し込んだカスタム attention で計算する。

```
prefix 行   : 通常の causal attention（fused SDPA、mask なし）
branch 行   : (a) 全 branch の query をまとめて shared prefix の key に attention  … (Σq) × S の 1 ブロック、mask なし
              (b) 各 branch が自分の key に causal attention  … branch を padding してバッチ化、(Q, qmax, qmax)
              (a)(b) を log-sum-exp で合成（flash attention と同じ恒等式）
```

attention の演算量は O(L²) から O(S²/2 + (Σq)·S + Σq²) になり、最大の一時領域は (rows × S) のブロックで、L×L も
Lc×(S+Lc) も作らない。fp32 で naive / packed と 1e-3 以内で一致し、isolation テストも通る。

(a) を fused SDPA で計算するには合成に必要な分配関数 Z = Σ exp(s) が要るが、SDPA は正規化後の出力しか返さない。
そこで **score 0 の zero key を 1 本足し、その value を probe（channel 0 だけ 1）にした 2 回目の SDPA** を呼ぶ。
probe channel の出力は c = 1/(Z + P)（P は padding した zero key の本数）なので Z = 1/c − P が得られる
（Qwen3 は q/k RMSNorm があり score は |s| < 50 程度で、c は fp32/bf16 とも underflow しない）。
`backend="manual"`（matmul + softmax 統計を chunk で計算）も残してあり、MPS では memory-bound で fused より遅い（S=8k・Q=1000 で 57 s vs 49 s 時点）。

**MPS での実測（上の表）**: 1 回の forward に収まる範囲では shared が packed より速い（S=8038・Q=100: 3.5 s vs 4.8 s、73x vs naive）。
Q=1000 の chunk 領域では packed（2048 token タイル）の方が速い（15.9 s vs 22.5 s）。理由は次の 2 点。

- MPS の SDPA は分配関数を出さないため、(a) に **2 回の全パス**が必要（probe 呼び出しは本体と同コスト）。
  head dim を 129/136 にして probe channel を足す方法は MPS では 10 倍遅くなる（`head_dim=128` 以外は遅いカーネル）ので使えない。
- packed の小タイルでは masked kernel の無駄が Lc/(S+Lc) ≈ 20% しかなく、mask 付きでも fused カーネル 1 回で済む。

つまり MPS では「物理的に sparsity を使う」効果より「fused カーネルの回数」が効く。CUDA では FlexAttention の `BlockMask`
（`backend="flex"`、本環境では未検証）で (a)(b) を 1 回の block-sparse カーネルにでき、この構造が本来の性能を出せるはず。
Jev 規模（state 23k × 5,000 問）では packed の L×L 相当（chunk でも Lc×(S+Lc)）は成立せず、D3 型の分解が必須になる。

## 設計メモ

- **prefix / suffix の分割 tokenize**: state 側と質問側を別々に tokenize し、全 engine が同じ token id 列を使う。
  境界で BPE の merge が起きないことをテストで確認している（`tests/test_prompt.py`）。
- **`kvcache` のメモリ**: HF の `DynamicCache` は batch 行ごとに prefix の K/V を実体化するため、
  長い state では batch を `cache_budget_bytes` で自動的に絞る。`packed` は prefix K/V を 1 部しか持たないのでこの問題がない。
  vLLM の prefix caching（paged KV）は前者を共有化する仕組みで、Linux/CUDA へ持っていくならそれで `kvcache` を置き換えられる。
- **長い packed 列**: `PackedEngine(max_tokens=…)` を超える場合は prefix を cache に入れ、質問側だけを chunk で pack する（同じ mask、query 行 = 質問のみ）。
- **明示 mask のコスト**: 4D mask を渡すと sdpa の causal 高速パスが使えず、8k 系列で約 2 倍遅くなる。
  そのため分岐が 1 本のとき（packed）と padding が不要なとき（naive）は mask を渡さない。
- **packed の block sparsity は演算削減に使われていない（reference 実装の限界）**: HF の SDPA / eager backend は
  `(1, 1, L, L)` の raw 4D mask を受け取り、attention を dense に計算してから mask する。共有されるのは prefix の
  線形層（QKV/MLP）の計算と K/V であり、attention 行列そのものは L² で作られる。
  S=8038・Q=100（L ≈ 13.5k）では理想比が線形層 60x・attention 36x で、実測 53x はその間に落ちる。
  Jev 規模（state 23k × 5,000 問、L ≈ 173k）では mask だけで bf16 60GB、attention FLOPs は理想の約 8 倍になり、
  この実装のままでは成立しない。物理的に sparsity を使う実装（shared prefix への attention を全 branch でまとめる
  Hydragen 型分解、CUDA では FlexAttention の BlockMask）は D3 として別に扱う。
- **bench.py は条件ごとに JSONL へ追記し、再実行時は完了済み条件をスキップする。** 途中で止めても失うのは実行中の 1 条件だけ。
- **校正**: 1-token readout の生 logits は非常に尖っており（MMLU で mean confidence ≈ 0.96, ECE ≈ 0.38）、
  temperature scaling だけで ECE は 0.06 前後まで落ちる。「0.8 と言ったら 8 割当たる」の最初の一歩はモデル改造なしで到達できる。

## 次フェーズ（未実装）

- **C: 専用 readout head** — `engine/head.py` で最終 hidden state → `nn.Linear(hidden_size, max_choices)`。LLM は freeze。
  語彙 15 万次元ではなく選択肢次元を直接出す。Hume の「slot head」相当。
- **E: 校正指向の post-training** — `L = CE + λ·Brier` で head または LoRA を学習（TRL/PEFT）。その先で RL / preference optimization。
- **本番サービング** — Linux GPU で vLLM + Automatic Prefix Caching、同じ `/decision` エンドポイント。
