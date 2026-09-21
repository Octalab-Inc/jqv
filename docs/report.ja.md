# jqv — Qwen ベース Decision API（Jev 簡易版）

全文の英訳は [report.md](report.md)、公開用の簡易 README は [../README.md](../README.md)。

TypeSafe の Jev（[Hume の推定](https://archerhume.com/posts/jevs-architecture-unmasked/?v=3)）と
PFN の Preference API / [Insight Scan](https://www.preferred.jp/ja/news/pr20250508) に共通する
「**生成せず、共有 state に対する多数の質問へ確率分布を直接返す**」構造を、オープン LLM（Qwen3）だけで再現する PoC。

```
POST /decision
{"state": "橋梁Aでは主桁に腐食が確認された…",
 "questions": [{"question": "主桁の損傷は？", "choices": ["損傷なし","腐食","ひび割れ"]}, ...]}
→ {"decisions": [{"probabilities": [0.01, 0.97, 0.02], "calibrated_probabilities": [...], "confidence": 0.9}, ...]}
```

## 結論（Findings）

1. **Jev の主要な推論挙動は普通の open decoder LLM で再現できる。** direct readout、shared state、sibling isolation（兄弟質問への漏れ 0、負対照 0.995）、
   listwise な選択肢相互作用（5 番目選択肢で log-odds が動く）を 1.7B / 14B / 32B で確認した。
2. **共有計算の高速化は本物で、「生成をやめる」こと自体はほとんど高速化にならない。** generate ≈ naive（0.9〜1.1 倍）に対し、
   長い state × 多数質問（S=8k・Q=100）では packed / shared が naive の 53〜73 倍（MPS 上）。
3. **Decision accuracy は今回の範囲では主に backbone capability で決まる。** 1.7B → 14B → 32B で MMLU 0.554 → 0.750 → 0.809、
   JevBench hard 0.423 → 0.550 → 0.622。4,800 例の LoRA は 14B 以上をほぼ改善せず、Jev との差は MMLU でも JevBench hard でも約 10〜12 ポイント残る。
4. **Calibration の「魔法」のかなりの部分は scalar correction でも再現できるが、domain shift は未解決。** 32B は MMLU で ECE 0.023（Jev の報告値 0.031）まで
   行く一方、同じ温度は JMMLU には転移し、JevBench hard には部分的（0.274 → 0.107）、bridge には逆効果と、タスクによって最適 scale が異なる。
5. **外部測定でも public の数値は再現され、Jev との差は縮まらない。** Benchmark Heaven（JevBench v1.2.7）は 32B zero-shot を
   easy 1.000 / standard 0.958 / judge 0.925 / hard public 0.622、Calibration 74.9、Score 67.2 と測定した（partial row、順位なし）。
   held-out hard 109 問は「提出者が運用する endpoint には送らない」方針で未測定。順位付きにするには serving code の公開か本番 endpoint が要る。

## 何を作ったか

同一モデル・同一プロンプトで、5 つの推論構造（generate / naive / kvcache / packed / shared）と、直交する 2 種類の readout（全語彙 projection / 選択肢行のみ）を切り替えて比較できる。

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

source の val で学習した T を、別の target の test にそのまま適用した。Jev の ECE 0.031（Hume の 1,200 問 MMLU サンプルでの測定。プロンプト条件は記事から確認できない）に対しては、
in-distribution の対角より非対角のほうが近い比較になる。

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

## C: 学習する decision head（`jqv/heads.py`, `jqv/train/`, `scripts/train_head.py`）

語彙 readout（B）の代わりに、最終 hidden state から直接 decision logits を出す head を学習する。プロンプトは B と同一。

| head | 式 | 特徴 |
|---|---|---|
| slot (C1) | `z = W h_d + b`（h_d は `Answer:` 位置の hidden state、行 = 選択肢スロット） | Hume の「slot head」。LM head の文字行で初期化すると step 0 は B' と同一 |
| pointer (C2) | `z_i = (U h_d)·(V h_i)/√r + w·h_i`（h_i は選択肢 i のテキスト末尾の hidden state） | K 可変、順序に等変（`tests/test_heads.py`）、任意ラベル。Hume の「pointer scorer」候補 |

- 学習データは MMLU `auxiliary_train`（99,842 問）。選択肢の並びを毎回 shuffle して位置 prior を学習させない。
  検証は MMLU `validation` から 256 問。評価は他と同じ MMLU / JMMLU の test 800 問。
- LLM は freeze し、LoRA（r=16, q/k/v/o_proj）を併用する設定と、LoRA なし（head だけ）の設定を比較する。
- 学習は MPS 上で `python -u scripts/train_head.py`。10 step ごとに loss / step/s / ETA を表示、100 step ごとに
  checkpoint（`results/train/<run>/last`）と検証（best は `best/`）、`--resume` で続きから再開できる。
- 推論は `make_engine("pointer", rt, head_dir="results/train/<run>/best")`。head は学習時の model と prompt_hash を記録し、不一致なら拒否する。

### 結果（600 step × batch 8 = 4,800 例、MPS で 11〜24 分、test 800 問。raw = 校正なし、+T = val 400 問で学習した温度）

| 方式 | 学習パラメータ | MMLU acc | MMLU NLL raw / +T | MMLU ECE raw / +T | JMMLU acc | JMMLU NLL raw / +T | JMMLU ECE raw / +T |
|---|---:|---:|---:|---:|---:|---:|---:|
| B: 語彙 readout（学習なし） | 0 | 0.554 | 5.79 / 1.08 | 0.41 / 0.080 | 0.466 | 6.13 / 1.19 | 0.49 / 0.066 |
| C1: slot + LoRA r=16 | 6.5M | **0.575** | 1.06 / **0.98** | 0.14 / 0.044 | **0.505** | 1.23 / **1.12** | 0.16 / **0.029** |
| C2: pointer + LoRA r=16 | 7.5M | 0.561 | 1.10 / 1.05 | **0.11** / 0.048 | 0.475 | 1.38 / 1.24 | 0.15 / 0.039 |
| C2: pointer、LLM 凍結 | 1.0M | 0.471 | 2.00 / 1.19 | 0.32 / 0.028 | 0.395 | 2.34 / 1.32 | 0.35 / 0.029 |

順序感度（MMLU 300 問 × 巡回シフト 4 通り、`--mode order`）:

| 方式 | 位置ごとの平均確率 1/2/3/4 | p(correct) の平均絶対差 | argmax 一致 |
|---|---|---:|---:|
| B | 0.25 / 0.31 / 0.22 / 0.22 | 0.220 | 48% |
| C1 slot + LoRA | 0.26 / 0.26 / 0.25 / 0.24 | 0.126 | 55% |
| C2 pointer + LoRA | 0.20 / 0.25 / 0.32 / 0.23 | 0.164 | 48% |

読み取れること:

1. **1.7B では decision training は生の校正を大幅に改善し、精度は JMMLU で有意に改善する（MMLU の +2.1 ポイントは有意でない）。** LoRA 併用の head は B より精度が高く（MMLU +2.1、p=0.125 / JMMLU +3.9、p=0.006。「精度の読み方」節）、14B 以上では accuracy の改善は消える（Backbone スケーリング節）。
   何より raw の確率が最初から使える（ECE 0.41 → 0.11〜0.14、NLL 5.8 → 1.1）。温度を足すと ECE 0.03〜0.05、NLL 0.98 に達し、
   B + 温度（0.080、1.08）を上回る。JMMLU の ECE 0.029 は Jev の MMLU 0.031 と同水準だが、こちらは in-distribution の温度込み。
2. **LLM を凍結して新規 head だけ学習すると B より悪い**（MMLU 0.471、−8 ポイント）。hidden state のどこに「選択肢 i の正しさ」が
   乗っているかを LLM は学習していないので、head だけでは読み出せない。LoRA で表現側を動かすことが必要。温度で ECE は直るが精度は戻らない。
3. **slot（LM head の文字行で初期化）≥ pointer（ゼロから学習）**。この学習量では B' から始められる slot が有利。
   pointer は式の上では順序に等変だが、h_i 自体が因果 attention で前の選択肢に依存するため順序効果は残る（argmax 一致 48%）。
   順序 shuffle で学習した slot は位置 prior がほぼ一様（0.26/0.26/0.25/0.24）になり、一致率も 48% → 55% に上がる。
4. 英語 MMLU で学習した head は日本語 JMMLU にも転移する（精度・NLL とも改善）。一方 bridge（日本語・読解型・30 問）では
   pointer + LoRA は 0.833 と B の 0.933 を下回る。学習分布外のタスク種別には注意が要る。
5. 学習は `results/train/<run>/train_log.jsonl` に step ごとの loss と検証値、`best/`・`last/` に checkpoint が残る。
   pointer + LoRA の best は step 300（val NLL 1.145）で、その後は過学習気味。

### E: 校正指向学習 `L = CE + λ·Brier`（slot + LoRA、λ ∈ {0, 0.5, 1, 2}、他は同一設定）

| λ | MMLU acc | NLL raw / +T | Brier raw / +T | ECE raw / +T | T | JMMLU acc | NLL raw / +T | ECE raw / +T | bridge NLL / ECE (T=1) |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0   | 0.575 | 1.056 / **0.977** | 0.554 / **0.523** | 0.137 / **0.044** | 1.82 | 0.505 | 1.233 / 1.118 | 0.158 / 0.029 | 0.064 / 0.056 |
| 0.5 | 0.573 | 1.052 / 0.979 | 0.551 / 0.524 | 0.135 / 0.051 | 1.78 | 0.505 | 1.220 / 1.119 | 0.150 / 0.036 | 0.057 / 0.050 |
| 1   | **0.579** | **1.049** / 0.981 | 0.552 / 0.524 | 0.128 / 0.051 | 1.76 | **0.507** | 1.217 / 1.119 | 0.147 / **0.028** | 0.061 / 0.053 |
| 2   | 0.549 | 1.069 / 1.043 | 0.571 / 0.560 | **0.082** / 0.046 | 1.47 | 0.480 | 1.217 / 1.171 | **0.098** / 0.044 | 0.147 / 0.120 |

- **λ ≤ 1 では CE 単独と区別がつかない。** accuracy / NLL / ECE の差は 800 問の信頼区間（±3.4 ポイント、ECE ±0.01 程度）の内側。
- **λ=2 は「生の確率」を平らにする方向に効く**（平均 confidence 0.71 → 0.63、raw ECE 0.137 → 0.082）が、精度が 3 ポイント落ち、
  温度を当てた後の NLL / Brier は λ=0 より悪い（0.977 → 1.043）。つまり Brier 項は識別力を上げるのではなく、学習時に温度を内蔵する働きをしている。
  校正セットが取れない運用では価値があるが、val で温度を 1 つ学習できるなら CE + T のほうが良い。
- **タスク種別をまたぐ転移はどの λ でも解けない。** bridge では全 λ で T=1 が最良で、MMLU の T を当てると ECE は 0.15〜0.19 に悪化する。
  λ=2 の bridge raw ECE（0.120）は λ=0（0.056）より悪い。
- Jev の ECE 0.031（Hume の測定、プロンプト条件は不明）に対し、slot + LoRA + in-distribution T は JMMLU で 0.028〜0.029、MMLU で 0.044〜0.051。
  「学習 + 温度」で数値上は同水準に届くが、温度なしで・分布をまたいで 0.03 を出す Jev の RLCD とは条件が違う。
- 1.7B では λ=1 が MMLU val NLL 最小（1.089 vs λ=0 の 1.103）だったが差はノイズ範囲。14B で再 sweep した結果 **λ=0** を採用し、32B も λ=0 とした（Backbone スケーリング節）。

## 学習なしの精度向上: few-shot state と選択肢の巡回シフト平均

目標は MMLU で 80% 前後。主レバーは backbone（14B / 32B、後述）だが、学習なしで効くものを 1.7B で先に確定した。
(a) **few-shot を shared state に置く**（packed なら prefill は subject ごとに 1 回で、質問あたりのコストは増えない）、
(b) **巡回シフト平均 `perm_avg`**（選択肢の並びを K 通り巡回して意味的選択肢に戻して確率を平均。位置・文字 prior を打ち消す）。
`scripts/eval.py --shots 5 [--shots-mode fixed] --perm-avg`、`make_engine(..., perm_avg=True)`。

| 設定（1.7B, packed, T=1 の精度） | MMLU acc | Δ vs zero-shot [95% CI] | p | ECE raw / +T | JMMLU acc | Δ | p | ECE raw / +T |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| zero-shot | 0.554 | - | - | 0.413 / 0.080 | 0.466 | - | - | 0.485 / 0.066 |
| 5-shot 同 subject（MMLU dev） | 0.550 | −0.004 [−0.036, +0.030] | 0.88 | 0.413 / 0.103 | 0.469 | +0.003 | 0.94 | 0.494 / 0.048 |
| 5-shot 固定（全質問共通の 5 例） | 0.422 | **−0.131** [−0.169, −0.094] | <0.001 | 0.546 / 0.047 | 0.421 | −0.045 | 0.014 | 0.529 / 0.061 |
| perm_avg（K=4） | **0.584** | **+0.030** [+0.001, +0.058] | 0.050 | **0.195** / **0.063** | 0.485 | +0.019 | 0.18 | 0.291 / 0.034 |
| 5-shot 同 subject + perm_avg | 0.578 | +0.024 [−0.009, +0.058] | 0.17 | 0.214 / 0.075 | **0.497** | +0.031 | 0.077 | 0.282 / 0.057 |

- **巡回シフト平均は学習なしで +3 ポイント**（MMLU、p=0.05）。生の ECE も 0.41 → 0.20 に下がり（平均 confidence 0.96 → 0.78）、温度も 12 → 8.8 に近づく。
  効果の源は順序実験で見た「argmax の 52% が並びで変わる」不安定さの平均化で、学習した slot + LoRA（0.575〜0.579）と同等の精度を推論だけで得る。
  校正後の選択的精度も改善する（p ≥ 0.7 で 27% を精度 0.89。zero-shot は 19% を 0.87）。
- **コストは 1.7 倍**（packed S=2038・Q=100: 48.6 → 28.1 q/s）。branch token は 4 倍だが state の prefill は共有されるため 4 倍にはならない。
- **同 subject の 5-shot は効かない**（±0.4 ポイント）。instruct モデルの直接 readout は書式を既に理解しており、例題は情報を足さない。
  JMMLU では英語例題を使ったが同じく効果なし。
- **全質問共通の固定 5 例は大きく害**（MMLU −13 ポイント）。予測文字が B に 64% 集中し（zero-shot 32%）、無関係な例題が文字 prior を誘導する。
  Jev 型の「1 つの state に多数の質問」で例題を共有するなら、例題の内容・正解文字の偏りが決定を歪めることを意識する必要がある。
- 14B / 32B へ持っていく設定: **zero-shot と perm_avg の 2 条件**。few-shot は同 subject 版のみ 1 回確認する。

### 精度の読み方（`scripts/compare_runs.py`）

精度は argmax の正解率で、温度では変わらない。Decision API では「確率が正直か」（ECE / NLL / Brier）と、
「高確信の問題だけ自動処理したときの精度と処理率」で読む。test 800 問の 95% 信頼区間は ±3.4 ポイントなので、
それ以下の差は同じ 800 問での対応比較（McNemar）で判断する。

#### mmlu test n=800: accuracy ±95% CI, paired Δ vs B (vocab) [bootstrap 95% CI], McNemar p
| run | accuracy | Δ vs baseline | discordant (win/lose) | McNemar p |
|---|---:|---:|---:|---:|
| B (vocab) | 0.554 ±0.034 | +0.000 [+0.000, +0.000] | 0/0 | 1.000 |
| slot+LoRA | 0.575 ±0.034 | +0.021 [-0.003, +0.048] | 63/46 | 0.125 |
| pointer+LoRA | 0.561 ±0.034 | +0.007 [-0.020, +0.037] | 77/71 | 0.681 |
| pointer frozen | 0.471 ±0.035 | -0.083 [-0.114, -0.051] | 49/115 | 0.000 |

#### selective accuracy with calibrated p (T from val): coverage / accuracy when p_max >= threshold
| run | p ≥ 0.5 | p ≥ 0.7 | p ≥ 0.9 |
|---|---|---|---|
| B (vocab) | 67% を 0.64 | 19% を 0.87 | 0% |
| slot+LoRA | 54% を 0.75 | 29% を 0.89 | 7% を 1.00 |
| pointer+LoRA | 52% を 0.74 | 24% を 0.83 | 3% を 0.88 |
| pointer frozen | 33% を 0.65 | 7% を 0.89 | 1% を 1.00 |

#### jmmlu test n=800: accuracy ±95% CI, paired Δ vs B (vocab) [bootstrap 95% CI], McNemar p
| run | accuracy | Δ vs baseline | discordant (win/lose) | McNemar p |
|---|---:|---:|---:|---:|
| B (vocab) | 0.466 ±0.035 | +0.000 [+0.000, +0.000] | 0/0 | 1.000 |
| slot+LoRA | 0.505 ±0.035 | +0.039 [+0.011, +0.068] | 75/44 | 0.006 |
| pointer+LoRA | 0.475 ±0.035 | +0.009 [-0.022, +0.040] | 86/79 | 0.641 |
| pointer frozen | 0.395 ±0.034 | -0.071 [-0.105, -0.037] | 70/127 | 0.000 |

#### selective accuracy with calibrated p (T from val): coverage / accuracy when p_max >= threshold
| run | p ≥ 0.5 | p ≥ 0.7 | p ≥ 0.9 |
|---|---|---|---|
| B (vocab) | 52% を 0.59 | 5% を 0.87 | 0% |
| slot+LoRA | 45% を 0.68 | 18% を 0.81 | 3% を 0.96 |
| pointer+LoRA | 32% を 0.63 | 8% を 0.70 | 0% を 0.00 |
| pointer frozen | 11% を 0.64 | 1% を 1.00 | 0% |

- 「slot + LoRA が B より高い」は JMMLU で有意（p=0.006）、MMLU では有意でない（p=0.125）。「pointer + LoRA は B と同等」「凍結 head は劣る」は確実。
- 全体精度が 2 ポイントしか違わなくても、校正後に p ≥ 0.7 で切ると slot + LoRA は B の 1.5 倍の問題（29% vs 19%）を同じ精度 0.89 で自動処理できる。
  大量分類で不確かなものだけ人に回す用途では、この「選択的精度」が効く。
- ここでの MMLU 0.55 は zero-shot・非 thinking・1,200 問サブセットの値で、公式評価（5-shot、全 14,042 問）とは比較できない。
  bridge は 30 問（1 問 = 3.3 ポイント）なので動作確認以上の意味はない。

## Backbone スケーリング（`scripts/scaling_table.py`, `results/scaling_table.md`）

同じコード・プロンプト・1,200 問サブセット（seed 0、val 400 / test 800）で backbone だけを変える。dense の Qwen3 に限定し、
MoE（30B-A3B）と Qwen3.5（hybrid attention）は attention / KV 構造の議論が変わるため使わない。Jev の行は Hume の測定値（1,200 問 MMLU サンプル、API が返した確率のまま。プロンプト条件は記事から確認できない）。

| backbone | params | B zero-shot: MMLU / JMMLU | B ECE raw / +T (T) | B + perm_avg: MMLU / JMMLU | perm_avg ECE raw / +T | 5-shot + perm_avg: MMLU | slot+LoRA: MMLU / JMMLU | slot ECE raw / +T | slot + perm_avg: MMLU | JevBench hard (public 111): B / perm_avg / slot | packed q/s (S=2k, Q=100) plain / perm_avg |
|---|---:|---:|---|---:|---|---:|---:|---|---:|---:|---:|
| qwen3-1.7b | 1.7B | 0.554 / 0.466 | 0.413 / 0.080 (12.0) | 0.584 / 0.485 | 0.195 / 0.063 | 0.578 | 0.575 / 0.505 (slot_lora) | 0.137 / 0.044 | - | 0.423 / 0.432 / 0.414 | 48.6 / 28.1 |
| qwen3-14b | 14.8B | 0.750 / 0.710 | 0.207 / 0.042 (5.1) | 0.781 / 0.729 | 0.113 / 0.047 | 0.782 | 0.757 / 0.711 (qwen3-14b_slot_brier0) | 0.114 / 0.045 | - | 0.550 / 0.568 / 0.586 | 9.3 / 4.4 |
| qwen3-32b | 32.8B | 0.809 / 0.771 | 0.137 / 0.023 (3.0) | 0.812 / 0.790 | 0.087 / 0.034 | - | 0.801 / 0.782 (qwen3-32b_slot_best) | 0.115 / 0.031 | 0.819 | 0.622 / 0.640 / 0.622 | 2.5 / - |
| Jev (TypeSafe, Hume 2025) | ? | **0.918** / - | 0.031 (Hume, 1,200 問, API の確率のまま) | - | - | - | - | - | - | **0.741** (534 決定、Benchmark Heaven) | 30k tok in ~160 ms |

### 1.7B → 14B で分かったこと

- **今回の範囲では精度を支配する最大の要因は backbone scale だった。** zero-shot の直接 readout で MMLU 0.554 → 0.750（+19.6）、JMMLU 0.466 → 0.710（+24.4）。
  1.7B で有効だった巡回シフト平均は 14B でも同じ幅で効き（+3.1、p=0.001）、5-shot は 14B では +1.9（p=0.12）、両方で 0.782（p=0.008）。
  学習なしの 14B で 0.78、Jev の 0.918 との差は 14 ポイント。
- **生の校正も backbone で良くなる。** 生 ECE 0.41 → 0.21、温度 12 → 5.1。温度後の ECE は 0.042（MMLU）/ 0.046（JMMLU）で、
  Jev の 0.031 との差は学習なしでも 0.01 台になった。巡回平均後の生 ECE は 0.113。
- **温度の転移は 1.7B と同じ構造。** MMLU↔JMMLU は転移する（T ≈ 5.0 で ECE 0.040〜0.048）。bridge は T=1 で ECE 0.000（30 問全問正解、confidence 0.99999）、
  MMLU の T を当てると 0.053 に悪化するが、1.7B の 0.24 より害は小さい。
- **共有計算の効果は backbone に依らない。** S=2038・Q=100 で packed は naive の 22 倍（1.7B は 20 倍）、generate ≈ naive（1.04 倍）も同じ。
  絶対速度は packed 9.3 q/s（1.7B の 48.6 q/s の 1/5.2、パラメータ比 8.7 倍より緩やか）。shared (D3) は S=8038・Q=100 で packed より 11% 速く、Q=1000 では遅い（1.7B と同傾向）。
- **isolation と選択肢相互作用も同じ。** packed / shared の兄弟質問への漏れ 0.0008、負対照 0.995。5 番目選択肢の Δlog-odds は bridge 21 問で
  −1.24 ± 0.42（1.7B は −0.49 ± 0.39）で、Hume が Jev で見た負方向のずれが 14B ではより強く出る。
- **等価性テストは fp32 で厳密に通り、bf16 では logit 差がちょうど 1 ulp（0.5）**。テストの許容差を dtype 依存にした。
- 14B の bench 所要時間: naive Q=100 で 238 s（1.7B 41 s）。以後 naive / generate は S=2000 までに限定する。

### 14B の slot + LoRA λ sweep（1.7B と同一設定: r=16、600 step × batch 8、選択肢 shuffle）

| λ | val NLL (best) | MMLU acc | Δ vs zero-shot [95% CI] | p | NLL raw / +T | ECE raw / +T | JMMLU acc | NLL raw / +T | ECE raw / +T |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| zero-shot B | - | 0.750 | - | - | 1.799 / 0.651 | 0.207 / 0.042 | 0.710 | 1.829 / 0.719 | 0.243 / 0.046 |
| perm_avg（参考） | - | **0.781** | +0.031 [+0.014, +0.049] | **0.001** | - / - | 0.113 / 0.047 | **0.729** | - / - | 0.141 / - |
| 0   | **0.675** | 0.757 | +0.007 [−0.011, +0.025] | 0.50 | 0.730 / **0.642** | 0.114 / 0.045 | 0.711 | 0.813 / **0.714** | 0.131 / 0.044 |
| 0.5 | 0.692 | 0.759 | +0.009 | 0.39 | 0.737 / 0.647 | 0.110 / 0.036 | 0.703 | 0.821 / 0.720 | 0.138 / 0.047 |
| 1   | 0.693 | 0.755 | +0.005 | 0.69 | 0.744 / 0.653 | 0.114 / 0.035 | 0.710 | 0.829 / 0.724 | 0.134 / 0.050 |
| 2   | 0.720 | 0.760 | +0.010 | 0.35 | 0.772 / 0.656 | 0.121 / **0.031** | 0.708 | 0.849 / 0.723 | 0.141 / 0.057 |

- **14B では 4,800 例の decision training は精度を上げない。** 4 水準とも +0.5〜1.0 ポイント（p ≥ 0.35）で、1.7B で見えた +2〜4 ポイントの効果は消える。
  同じ backbone で perm_avg は +3.1（p=0.001）なので、学習より推論側の順序平均のほうが効く。
- **生の校正は学習で改善する**（ECE 0.21 → 0.11〜0.12、NLL 1.80 → 0.73〜0.77）が、温度を当てた後は zero-shot + T（NLL 0.651、ECE 0.042）と同等。
  λ=2 の MMLU ECE+T 0.031 は Jev の 0.031 と同じ値だが、JMMLU では 0.057 で最悪。λ 間の差はノイズ範囲。
- 1.7B で見えた「λ=2 で精度が落ちる」現象は 14B では起きない（0.760）。
- 32B に持っていく設定は **λ=0**（MMLU val NLL 最小 0.675。`results/scaling_best_config.json`）。1.7B の選択は λ=1 だったが、どちらも差はノイズで、CE 単独が最も単純。
- 学習は 1 run 97〜119 分（0.09〜0.10 step/s、`--grad-accum 2`）、評価は 1 データセット 9〜12 分。`--resume` は λ=1 の run で 20 step → 600 step の再開を確認。

### 32B: 最終確認（B、perm_avg、slot + LoRA λ=0、slot + LoRA + perm_avg）

| 32B（test 800） | MMLU acc | Δ vs zero-shot [95% CI] | p | NLL raw / +T | ECE raw / +T | T | JMMLU acc | ECE raw / +T |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| zero-shot B | 0.809 | - | - | 0.947 / 0.535 | 0.137 / **0.023** | 3.0 | 0.771 | 0.158 / 0.041 |
| B + perm_avg | 0.812 | +0.004 [−0.013, +0.020] | 0.76 | 0.740 / 0.528 | 0.087 / 0.034 | 2.4 | **0.790** | 0.078 / **0.018** |
| slot + LoRA λ=0 | 0.801 | −0.007 [−0.021, +0.006] | 0.38 | 0.674 / 0.550 | 0.115 / 0.031 | 1.7 | 0.782 | 0.109 / 0.040 |
| slot + LoRA + perm_avg | **0.819** | +0.010 [−0.007, +0.028] | 0.34 | 0.583 / 0.533 | 0.063 / 0.042 | 1.4 | - | - |

学習: 600 step × batch 8（micro batch 2 × 累積 4、gradient checkpointing）で 302 分（0.03 step/s）。メモリは 123〜127 GB 使用で 128 GB に収まった。
best は step 300（val NLL 0.688）。評価は slot engine で 1 データセット 22〜24 分、perm_avg 付きで 60 分。

### 1.7B → 14B → 32B の結論

1. **今回試した 4,800 例規模の decision training に比べ、精度を支配する最大の要因は backbone scale だった。** zero-shot の直接 readout で MMLU 0.554 → 0.750（+19.6）→ 0.809（+5.9）、Jev（0.918）との差は 36 → 17 → 11 ポイント。残る約 11 ポイントは backbone 自体の能力差か、Jev の大規模 post-training の効果と考えられる。
   目標の 80% は 32B の zero-shot で到達した。推論側の工夫（perm_avg）は 1.7B / 14B で +3 ポイント、32B では +0.4 に縮む。
2. **MMLU 系の in- / near-distribution では scalar temperature だけで Jev の報告 ECE と同水準に達する。一方その温度は任意のタスクへ普遍的には転移しない。** ECE+T は 0.080 → 0.042 → 0.023（Jev 0.031）。温度は 12 → 5.1 → 3.0 と単調に下がり、
   JMMLU で学習した T=2.6 を MMLU に当てても 0.031。ただし同じ T は JevBench hard では部分的にしか効かず（0.274 → 0.107）、bridge では逆効果になる。
   ただし Jev は温度なしの値で、jqv の生 ECE は 32B でも 0.137（perm_avg 後 0.087）。
3. **4,800 例の decision training は 14B 以上で精度を上げない。** slot + LoRA は 1.7B で +2〜4 ポイント、14B で +0.5〜1.0（有意差なし）、
   32B で −0.7（p=0.38）。生の校正は改善する（ECE 0.137 → 0.115、NLL 0.95 → 0.67）が、温度後は zero-shot + T を超えない。
   Jev との残り 10 ポイントは、この規模の head 学習では埋まらない。backbone 能力か、桁違いに大きい post-training（RLCD）の効果と考えるのが自然。
4. **共有計算の効果と isolation は backbone に依らない。** Q=10 で共有系 3 engine は naive の 7〜9 倍、漏れは 3 サイズとも ≈0 / 負対照 0.995。
   絶対速度は packed で 48.6 → 9.3 → 2.5 q/s（S=2k・Q=100）で、パラメータ数にほぼ比例して遅くなる。
5. **mask なしの D3 は backbone が大きいほど有利。** S=2038・Q=100 で shared / packed は 1.7B 0.87 倍、14B 1.07 倍、32B **1.50 倍**。
   層数と head 数が増えるほど dense masked attention の無駄が効く。
6. **選択的精度（32B、校正後）**: p ≥ 0.9 で MMLU の 44% を精度 0.97、p ≥ 0.7 で 75% を 0.91 で自動処理できる（slot + perm_avg は 50% を 0.97）。
7. 5 番目選択肢の Δlog-odds は 1.7B −0.49、14B −1.24、32B +0.08（いずれも bridge 21 問、CI ±0.4）で一貫せず、n=300 での再測定が必要。

次の外部評価は JevBench（Benchmark Heaven、Jev の hard tier 74.1%）で、MMLU の差 11 ポイントが Decision 用途でどうなるかを見る（`tasks/active/jevbench-eval.md`）。

## JevBench での外部評価（Benchmark Heaven v1.2、public 231 決定）

[JevBench](https://github.com/fstandhartinger/jevbench)（MIT）は Jev 系 decision model 専用のベンチマークで、state + rubric に対する
typed decision を choice / noul / score の 3 型で問う。public 分は easy 48 / standard 72 / hard 111 = 231 決定（judge 146 と held-out 303 は非公開で、
[Benchmark Heaven](https://benchmarkheaven.com/jev-models) が bench request で測る）。hard は 2〜6k token の policy、multi-hop、
日付・数値推論、adversarial distractor、「明確な答えなし」が正解の問題を含み、MMLU より Decision API 本来の用途に近い。

jqv は TypeSafe 互換の `POST /v1/systemone`（`jqv/systemone.py`）を実装し、ハーネスの `typesafe` adapter を改変なしで使った
（`scripts/jevbench_run.py`、1 リクエストずつ、ローカル MPS、ハーネス commit 7ce310c）。JSON state（public の 35 件）は整形 JSON テキストとして与える。
温度は MMLU val で学習した T をそのまま転用し、**サーバが校正済み確率を返す構成**（`jevbench_run.py --temperature-file`）で 9 設定を
ハーネスに測らせた。生の確率を返す構成の 9 設定も走らせてあり（`results/jevbench/<label>/`、`_T` なし）、精度は両者で同一、
ハーネスが測った校正済み ECE は生確率に T を事後適用した値と一致する（`results/jevbench_summary.md` に両方の行がある）。

| run（サーバが校正済み確率を返す構成） | easy 48 | standard 72 | hard 111 | Intelligence（public 3 tier） | hard ECE | Calibration（ECE のみ） | 生 p50 |
|---|---:|---:|---:|---:|---:|---:|---:|
| qwen3-1.7b zero-shot (T=12.0) | 1.000 | 0.625 | **0.423** | 61.4 | 0.151 | 69.8 | 0.04 s |
| qwen3-1.7b perm_avg (T=8.8) | 1.000 | 0.708 | **0.432** | 65.0 | 0.139 | 72.3 | 0.07 s |
| qwen3-1.7b slot + LoRA (T=1.8) | 1.000 | 0.681 | **0.414** | 63.2 | 0.146 | 70.9 | 0.05 s |
| qwen3-14b zero-shot (T=5.1) | 1.000 | 0.875 | **0.550** | 76.4 | 0.126 | 74.9 | 0.31 s |
| qwen3-14b perm_avg (T=3.9) | 1.000 | 0.889 | **0.568** | 77.7 | 0.185 | 62.9 | 0.53 s |
| qwen3-14b slot + LoRA (T=1.6) | 1.000 | 0.889 | **0.586** | 78.4 | 0.193 | 61.3 | 0.30 s |
| qwen3-32b zero-shot (T=3.0) | 1.000 | 0.958 | **0.622** | 82.6 | 0.107 | 78.6 | 0.65 s |
| qwen3-32b perm_avg (T=2.4) | 1.000 | 0.944 | **0.640** | 82.8 | 0.115 | 77.0 | 1.37 s |
| qwen3-32b slot + LoRA (T=1.7) | 1.000 | 0.944 | **0.622** | 82.1 | 0.132 | 73.6 | 0.80 s |
| Jev 1.13（Benchmark Heaven v1.2.5、534 決定、held-out 込み） | - | - | **0.741** | 90.4 | - | 82.7（ECE + TVD） | 0.65 s |
| SemIf Qwen3.5-4B（同） | - | - | 0.595 | 85.9 | - | 72.6 | 0.20 s |
| openjev-sglang Qwen3.6-35B-A3B（同） | - | - | 0.714 | 88.9 | - | - | 0.68 s |
| GPT-5.6 Luna low（同） | - | - | 0.945 | 96.8 | - | 89.8 | 0.97 s |

Intelligence は public 3 tier の重み付き精度（easy 14 / standard 28 / hard 30、judge 欠損分は正規化）で、リーダーボードの値（judge + held-out 込み）とは
条件が異なる。Calibration は hard tier の top-label ECE のみ（分布正解 20 問は非公開）。latency は補正なしの生値
（Benchmark Heaven は self-hosted に ×2 + 0.15 s を加える）。

32B zero-shot の hard tier を family 別に見ると（正解 / 問題数）: adversarial 6/6、trap 8/8、routing 5/5、judge 14/17 に対し、
long_policy 9/19（perm_avg で 11/19）、multi_hop 10/18、temporal_numeric 5/15、probability 4/10、tradeoff 3/6。

読み取れること:

1. **easy は 1.7B でも 100%、standard は 32B で 95.8%。** 単純な分類 / routing / 明示的な答えの抽出は小さい backbone で飽和しており、
   backbone 差が出るのは hard（1.7B 42.3 → 14B 55.0 → 32B 62.2%）。
2. **Jev との差は MMLU と同じ幅。** hard tier で Jev 74.1% に対し jqv-32B は 62.2%（perm_avg 64.0%）で 10〜12 ポイント差、
   MMLU の差（10.9 ポイント）とほぼ同じ。Decision 用途で差が縮むことも広がることもない。SemIf Qwen3.5-4B（59.5%）は上回り、
   openjev-sglang Qwen3.6-35B-A3B（71.4%）には届かない。ただし Jev 側の値は held-out を含む 220 問での値。
3. **弱いのは長い規則の適用と数値・時間推論**（long_policy 47〜58%、temporal_numeric 33%、probability 40〜50%）。
   Benchmark Heaven の topic 別で Jev が open model に大きく差をつけるのも rules / policy / finance で、Jev の post-training が
   「規則を読んで typed decision をする」方向に効いている可能性と整合する。逆に adversarial / trap / routing は 32B で満点。
4. **生の確率のままでは Calibration 軸が壊滅する。** hard の生 ECE は 32B でも 0.27（1.7B 0.54）で、Calibration 換算 45（1.7B は 0）。
   MMLU val の T をサーバに載せて返すだけで、ハーネス計測の ECE は 0.107、Calibration 78.6（Jev 82.7、SemIf 72.6）まで回復する。
   Decision API は校正済み確率を返して初めて評価に乗る。MMLU → JevBench hard への温度転移は**部分的**: ECE は 0.274 → 0.107 と大きく改善するが、MMLU 内の 0.023 には遠く、転移節で定義した基準（対角 + 0.02 以内）は満たさない。MMLU ↔ JMMLU（強く転移）、MMLU → JevBench hard（部分転移）、MMLU → bridge（逆効果）という 3 段階の distribution shift が見える。
5. **decision training（slot + LoRA）はここでも効かない。** 32B で hard 62.2%（zero-shot と同じ）、14B で 58.6%（+3.6）。
   perm_avg は 32B hard で +1.8。
6. **速度**: 32B の生 p50 は 0.68 s（Jev 0.65 s、ただし Jev は本番 API）。Benchmark Heaven の補正（×2 + 0.15 s）を当てると 1.5 s 相当。
   1.7B は 0.04 s。

但し書き: public 231 問のみ（judge tier と分布正解問題なし）、英語のみ、hard は Claude Opus 5 / GPT-5.6 Sol 作成の pilot。
Jev 行の数値は Benchmark Heaven の測定値（全 534 問）を引用したもので、同一問題での対応比較ではない。

### Benchmark Heaven による測定（JevBench v1.2.7、2026-09-21）

上の公開 endpoint を bench request（[jevbench#6](https://github.com/fstandhartinger/jevbench/issues/6)）で提出し、Benchmark Heaven がドイツから
`typesafe` adapter 無改変・1 リクエストずつで測定した。結果は [JevBench v1.2.7](https://github.com/fstandhartinger/jevbench/blob/v1.2.7/RESULTS-v1.2.md) と
[benchmarkheaven.com/jev-models](https://benchmarkheaven.com/jev-models) に **partial row（表示はされるが順位なし）** として掲載された。
測定条件と価格基準は実行前にコミットされた [`docs/v1.2-additions-run3.md`](https://github.com/fstandhartinger/jevbench/blob/v1.2.7/docs/v1.2-additions-run3.md) にある
（公開された行の JSON と合わせてコピーを `results/jevbench/published_v1.2.7/` に置いた。maintainer のコメント全文は `results/public/jevbench_issue6_comments.md`）。

| 項目 | Benchmark Heaven の測定値 | 自前のハーネス測定（public のみ） |
|---|---:|---:|
| easy（72 問、held-out 込み） | 72/72 = 1.000 | 1.000（48 問） |
| standard（96 問、held-out 込み） | 92/96 = 0.958 | 0.958（72 問） |
| judge（146 問、非公開） | 135/146 = 0.925 | 未測定 |
| hard public（111 問） | 69/111 = 0.622 | 0.622 |
| hard held-out（109 問） | 送信されず、未回答扱い | 非公開 |
| hard tier（220 問として集計） | 0.314 | - |
| hard ECE / Brier | 0.107 / 0.515 | 0.107 / - |
| Calibration 軸 | 74.9（ECE に加えて probability fidelity 71.1） | 78.6（ECE のみ） |
| p50 / p95（standard + judge、ドイツから、生値） | 0.92 s / 4.71 s | 0.65 s（ローカル） |
| Speed 軸 | 67.6（self-hosted 補正 ×2 で p50 1.85 s 相当） | - |
| Cost 軸 | 52.8（$0.0374 / 1,000 決定、OpenRouter `qwen/qwen3-32b` の入力単価で推定） | - |
| JevBench Score | **67.2**（Intelligence 76.1、順位なし） | - |

読み取れること:

1. **public の自己測定値は完全に再現された**（easy 1.000、standard 0.958、hard 0.622）。サーバが校正済み確率を返す構成、prompt hash、温度は
   `/health` の応答で先方に確認されている。
2. **judge tier（非公開 146 問）は 92.5%。** 同じ tier で Jev 1.13 は 94.5%、classifier.dev（Jev のバッチ）は 97.3%。standard と同様に backbone 差が小さい領域。
3. **hard は held-out 109 問が送られず、row は 534 決定中 425 で順位なし。** 測定の途中で「提出者が運用する endpoint には held-out を送らない」方針が
   採られたため（easy / standard / judge は方針決定前に送信済み）。順位付きの row にするには公開 weights か serving code、または本番 endpoint が要る
   （maintainer のコメント）。現状はリポジトリ非公開のため partial のまま。
4. **Calibration 74.9 は ECE だけの自己測定 78.6 より低い。** Benchmark Heaven は分布正解問題の probability fidelity（71.1）も含める。
   Jev 82.7 との差は主にここに出る。
5. **Speed と Cost は評価条件の影響が大きい。** 生 p50 0.92 s はローカル 0.65 s に日独往復が乗った値で、self-hosted 補正 ×2 で 1.85 s 相当。
   Cost は無料 endpoint でも base model の公開単価で見積もられる。Speed 軸まで競うなら欧州近傍の CUDA サーバに置く。

## 公開 endpoint（Cloudflare Quick Tunnel）

held-out 303 問と judge tier は Benchmark Heaven が bench request で測る。jqv の `/v1/systemone` は TypeSafe 互換なので、
HTTPS で公開すればハーネスの `typesafe` adapter（`--key-env ''`）からそのまま測定対象になる。一時公開の手順:

```bash
brew install cloudflared
caffeinate -dimsu &                                                    # 測定中はスリープさせない
JQV_MODEL=Qwen/Qwen3-32B JQV_ENGINE=packed \
JQV_TEMPERATURE_FILE=results/mmlu_packed_qwen3-32b_temperature.json \
uv run uvicorn jqv.server:app --host 127.0.0.1 --port 8000 &          # 校正済み確率を返す構成
cloudflared tunnel --url http://localhost:8000                          # https://<random>.trycloudflare.com が発行される
curl -s https://<random>.trycloudflare.com/health                       # calibration.temperature が出ることを確認
```

- Quick Tunnel はアカウント・DNS 設定不要、URL はランダムで `cloudflared` を止めると消える。公開中は URL を知る誰でも叩けるので、測定後すぐ停止する。
- Benchmark Heaven はドイツから 1 リクエストずつ呼ぶため、外部の p50 にはネットワーク遅延が乗る（ランキングでは self-hosted に ×2 + 0.15 s の補正が入る）。
  同じ Mac から公開 URL を叩いた実測では、ローカル 0.23〜0.74 s のリクエストがトンネル経由で +0.04〜0.12 s（日本国内の Cloudflare edge 経由）。ドイツからはさらに往復分が乗る。Speed 軸まで競うなら欧州近傍の Linux/CUDA に置く。
- 提出するのは 32B zero-shot + T=3.0（hard 0.622、Calibration 78.6、p50 0.65 s）。perm_avg は +1.8 pt に対し p50 が 2 倍でスコア上不利。
- bench request は 2026-09-21 に提出し（[fstandhartinger/jevbench#6](https://github.com/fstandhartinger/jevbench/issues/6)、32B zero-shot + T=3.02、リポジトリは非公開のまま。提出文は `results/public/bench_request_issue.md`）、同日午前に測定され JevBench v1.2.7 に掲載された（上の節）。「測定完了、トンネルを落としてよい」のコメントを受けて 08:51 に停止。cloudflared のカウンタでは公開中の総リクエスト 541（200 が 511、400 が 2 は自分の probe、404 が 27 はパス探索）。
- 2026-09-21: serving code を [Octalab-Inc/jqv](https://github.com/Octalab-Inc/jqv) として公開し、held-out を含む順位付きの再測定を [jevbench#9](https://github.com/fstandhartinger/jevbench/issues/9) で依頼した（maintainer 側が自分のハードで実行する。手順は `docs/jevbench-serving.md`）。
- 公開中の監視は `scripts/watch_public.sh`（issue のコメント、トンネル経由と localhost の health、cloudflared `/metrics` のリクエスト増分を 5 分ごとに `results/public/watch_public.log` に記録し、消えたプロセスは再起動する。GitHub には書き込まない）。uvicorn のアクセスログを git 管理下の `results/public/server.log` に向けていたため、ブランチ切替でファイルが差し替わり測定中のログを失った。`results/public/*.log` は ignore にした。

## 弱い hard family 向けの合成データ（`jqv/synth/`, `data/synth/`）

32B が JevBench で最も弱い family は long_policy（9/19）、temporal_numeric（5/15）、probability（4/10）。targeted training には
正解が確実なデータが要るので、合成データの正解はすべて solver（rule engine、`datetime` / `zoneinfo` による暦計算、`fractions` による
厳密な確率）で生成し、LLM はナラティブ段落の言い換えにだけ使う。item は JevBench と同じ形（state、`choice` / `noul` / `score` の
typed question と criteria、labels、expected）で、`data/synth/<family>/{train,dev,test}.jsonl`（family ごとに 2,000 / 300 / 500。
dev と test はコミット、train は seed 0 と保存した paraphrase patch から再生成）に置く。

- **long_policy**: rule engine を持つ 5 ドメイン（住宅の水損、商用設備の故障、旅行キャンセル、人事の転居費用規程、SLA クレジット）。
  各文書（1.3〜3.0k token、中央値 2.25k）は定義、例外付きの番号付き除外条項、サブリミットや閾値を変える発効日付きの特約・改定、
  一般条項、往復文書の抜粋、請求ファイルを持ち、表面的な答えに誘導する研修生メモを含む。決定ラベルは同じ規則を実行して決める。
- **temporal_numeric**: 6 シナリオ（月末規則・うるう年と時差、祝日と時間外受付を含む営業日期限、休職リセット付きの継続勤務月数、
  日割り請求、DST の締切、上限に対する単位換算）。state 内のメモが誤った計算を示す。
- **probability**: 6 シナリオ（旧版の計画が残る抜取検査の超幾何確率、年齢層別有病率からの事後確率、k-of-n 冗長系、期待値による選択、
  供給元の混合とベイズ、抽出の結果）。確率的事象を問う `noul` は真の P(yes) を `target_distribution` に、抽出結果の `choice` は
  結果分布そのものを持ち、後続の proper scoring 実験に使う。
- **multi_hop は family ではなく属性**: solver の導出トレースから全問に `dependency_hops`（導出した中間事実の数）と
  `reasoning_depth`（最長の導出連鎖）を付ける。hops は 2〜7。
- **誤誘導**: メモの近道が正解と違う答えになる問題を優先して生成する（dev で temporal_numeric 60%、probability 54%）。
  JevBench hard が問うている失敗様式そのもの。
- **汚染**: JevBench public 231 問との単語 8-gram 一致 0、ID・固有名の再利用なし（`scripts/synth_contamination.py`）。
  最初の草稿が JevBench の例から引きずっていた定型句とラベル語彙を書き換えて到達した。

難易度確認（`scripts/synth_difficulty.py`, `results/synth_difficulty.md`。zero-shot、packed、family ごとに dev 300 問）:

| family | 14B dev | 32B dev | JevBench 32B（目標 ±10 pt） |
|---|---:|---:|---:|
| long_policy | 0.333 | 0.383 | 0.47（9/19） |
| temporal_numeric | 0.270 | 0.323 | 0.33（5/15） |
| probability | 0.447 | 0.453 | 0.40（4/10） |

初回生成は temporal_numeric（14B 0.507）と probability（14B 0.657）が易しすぎた。難化を 2 回（正解と食い違う誤誘導、2 択の削減、
scenario 内での引き直し、parameter 空間の拡大）行い、32B で 3 family とも目標範囲に入った。誤誘導メモが正解と食い違う問題では
モデルはほぼメモに従う（temporal_numeric の surface-answer rate 0.6〜1.0）。精度は hops に対して単調ではなく、long_policy では
6 hop の問題（「サブリミット内で支払う」決定）が最も易しい。hops は導出の長さであって難しさそのものではない。表現の多様化: Qwen3-14B が train の各 item で事実の記述段落を 1 つ言い換えた（請求ファイル・報告・記録のみ。規則、条項、誤誘導メモは書き換えない）。数値・日付・ID・固有名がすべて残り、長さが 0.6〜1.6 倍で、比較語と否定語の出現数が変わらない場合だけ受理する。2 時間の枠内で long_policy 612 件（31%）、temporal_numeric 568 件（28%）、probability 125 件（6%）の train item が変わった。受理分の約半数はモデルが原文をそのまま返したもので、それらは除いた。受理した書き換えは `train.paraphrase.jsonl` の patch として保存し、train の再生成時に再適用する。dev と test は変更しない。

テスト: `tests/test_synth.py`（25 件）が、各 scenario の `facts` 上書きによる手計算ケース、トレースの集計、item の不変条件、
split の重複排除、loader の往復を検査する。

## 関連プロジェクト

同じ仮説（生成せず選択肢 token の logits を直接読む、shared state を 1 回 prefill する、学習 head、Brier 学習、shared-prefix attention）に
2026 年に複数の公開実装が独立に到達している。URL は確認済み（2026-09-20）。数値は各リポジトリの README の自己申告で、jqv とは benchmark も条件も異なる。

| project | jqv との近さ | 特徴 | jqv との差 |
|---|---|---|---|
| [featherless-ai/simple-jev](https://github.com/featherless-ai/simple-jev) | ★★★★★ | prefill の next-token logits で answer label を読む。共通 prefix の KV cache を質問 suffix のバッチで再利用。RFDT（answer-token logits を直接学習、LoRA） | jqv の naive → kvcache（B → D1）と同じ発想。README で「softmax 値は校正された正解確率ではない」と明記。jqv は packed / shared（D2 / D3）と校正の測定まで進めている |
| [TianyuCodings/NanoJev](https://github.com/TianyuCodings/NanoJev) | ★★★★★ | Qwen3-0.6B + 専用 decision head。CE / Brier 学習、observed-event probability、paired proper-reward、RLCD 風実験。simulator（maze / Snake）で真の確率 q と比較 | jqv の C / E と同テーマ。NanoJev は真値 q が取れる制御環境、jqv は MMLU / JMMLU / bridge の一般 semantic decision。「ECE だけでなく NLL / Brier、risk-coverage、OOD を分けて見る」という評価方針は jqv の `compare_runs.py`（選択的精度）と温度転移表に対応する |
| [TheoLeeCJ/SemIf](https://github.com/TheoLeeCJ/SemIf)（旧 OpenJev） | ★★★★☆ | Qwen3.5-4B、runtime-defined criteria、direct option logits、shared-state prefill。自作 benchmark と TypeSafe 公開 subset で比較（README の自己申告値） | 「scores は Jev 型の運用校正ではない」と明記し、issue で「task ごとに最適温度が違うのでは」が議論されている。jqv はこれを実測済み（言語間は転移、task 種別は転移しない） |
| [ekzhang/openjev-sglang](https://github.com/ekzhang/openjev-sglang) | ★★★★☆ | Qwen3.6-35B-A3B + SGLang radix cache で Jev 互換 API（choice / noul / rubric、prefill のみ）。TypeSafe 公式 SDK で疎通 | jqv の「本番サービング」フェーズを先行。校正は「supplied options に条件付いた確率で、正解の校正推定ではない」と明記 |
| [r-ms/mini-jev](https://github.com/r-ms/mini-jev) | ★★★★☆ | 凍結 Qwen3-4B で option letter の logits を読む事前登録実験 | jqv の B と同じ readout。学習・共有計算・校正は扱わない |
| [zwliJay/jev-forge](https://github.com/zwliJay/jev-forge) | ★★★☆☆ | shared prefix から dynamic candidate branch をスコアする学習・推論スタック（高 cardinality、校正、バッチ推論） | 学習スタックとしては近い。jqv は engine の ablation と校正測定に寄っている |
| [bnsd55/jevmlx](https://github.com/bnsd55/jevmlx) | ★★★★☆ | Apple Silicon / MLX 向けの実用 Decision API。context + schema catalog を 1 回 prefill し、各 field の選択肢を broadcast KV cache に対する trie 行として 1 回のバッチ forward で評価。schema 制約付き JSON、CalibrationBundle、timing ledger | 実行構造は jqv の D1（shared KV）寄り。structured prediction 側（schema、trie、multi-field）は jqv より作り込まれている。D2 / D3 の architecture ablation は扱わない |
| [Hydragen](https://arxiv.org/abs/2402.05099)（Juravsky et al., 2024） | システム | shared prefix への attention を全 suffix の query でまとめて計算し、KV の読み出しを共有。CodeLlama-13B で最大 32x | jqv の D3（`jqv/engine/shared.py`）はこの分解の PyTorch 実装。MPS では probe 用の 2 回目 SDPA が要るため dense packed に負ける条件がある |
| [DeFT](https://arxiv.org/abs/2404.00242)（Yao et al., ICLR 2025） | システム | tree 構造の推論向け Flash Tree-attention。shared KV の IO を 73〜99% 削減 | jqv の packed / D3 の CUDA 実装（FlexAttention BlockMask）の先にある方向 |

他に同種の実装として [cobanov/awesome-jev](https://github.com/cobanov/awesome-jev)（Jev 関連プロジェクトの一覧）、
[rongxinzy/LightJev](https://github.com/rongxinzy/LightJev)、[shamazharikh/qwen-rlcd](https://github.com/shamazharikh/qwen-rlcd) がある。

**アプリケーション / downstream**（再実装ではなく Jev を使う側）: [classifier.dev](https://classifier.dev/)（[mrmps/classifier-dev](https://github.com/mrmps/classifier-dev)）は TypeSafe Jev を backbone に大量分類を HTTP / CLI / MCP で提供し、Jev が使えないときは LLM に fallback する（最大 20 決定）。jqv の `/v1/systemone` はこの種のクライアントからも同じ wire format で呼べる。

jqv の独自性は次の 4 点にある。

1. engine を A / B / B' / D1 / D2 / D3 に分解し、fp32 で choice logits が一致することをテストで担保したうえで速度を比較している（他は shared KV までで、packed block mask と causal 負対照の系統比較はない）。
2. Hume の secret-code isolation を負対照付きで再現している（packed 0.000、packed_causal 0.996）。
3. 一般 semantic benchmark（MMLU / JMMLU / bridge）で raw → 温度 → slot + LoRA → CE + λ·Brier を同一 test で比較し、対応比較と選択的精度で報告している。
4. 温度の domain 転移（言語間は転移、task 種別は転移しない）を実測している。

一方で、NanoJev が RL を「sampling / interaction しか得られない場合の手段」と整理し、logits と真値が取れるなら Brier を直接 backprop する方が
自然としている点は、jqv の E（CE + λ·Brier を先に試す）の方針と一致する。jqv の結果では λ ≤ 1 は CE と区別がつかず、分布をまたぐ校正は
温度でも Brier でも解けていないので、次は RLCD 型（結果ベースの proper scoring）が候補になる。

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

## 今後の課題

- **RLCD / proper scoring の本格評価** — CE + λ·Brier では temperature scaling を超えなかった（E 節）。分布をまたいで校正を保つには結果ベースの proper scoring による post-training が次の候補。
- **CUDA / FlexAttention / Hydragen 型のサービング** — D3 は MPS では probe 用に SDPA が 2 回必要で、32B で packed の 1.5 倍にとどまる。CUDA では block-sparse カーネル 1 回にできる。vLLM / SGLang の prefix caching 上で同じ `/decision`・`/v1/systemone` を出す。
- **domain-shift calibration** — MMLU ↔ JMMLU は転移、JevBench hard は部分転移、bridge は逆効果。タスク種別ごとの温度、または温度に依存しない校正学習。
- **JevBench の順位付き row** — v1.2.7 の row は held-out hard 109 問が未送信で順位なし（judge tier は測定済み）。順位を得るには serving code の公開（リポジトリ公開）か本番 endpoint が要る。弱い family（long_policy、temporal_numeric）への few-shot / perm_avg の限定適用。
- **実アプリケーション** — classifier.dev 型の大量分類、grep 型のルーティングを `/v1/systemone` の上に載せて、選択的精度（p ≥ 0.9 で 44% を精度 0.97）を運用指標として使う。
