# jqv — full experimental report

The complete write-up of the experiments behind [jqv](../README.md) (English translation of the original Japanese report, [report.ja.md](report.ja.md)). Paths are relative to the repository root. Run instructions for JevBench reviewers: [jevbench-serving.md](jevbench-serving.md).

A proof of concept that reproduces, with nothing but an open LLM (Qwen3), the structure shared by TypeSafe's Jev
([Hume's reconstruction](https://archerhume.com/posts/jevs-architecture-unmasked/?v=3)) and PFN's Preference API /
[Insight Scan](https://www.preferred.jp/ja/news/pr20250508): **no generation; many questions against one shared state; a
probability distribution returned directly for each question.**

```
POST /decision
{"state": "Bridge A: extensive corrosion on the lower flange of the main girder ...",
 "questions": [{"question": "What damage does the main girder show?", "choices": ["none", "corrosion", "cracking"]}, ...]}
→ {"decisions": [{"probabilities": [0.01, 0.97, 0.02], "calibrated_probabilities": [...], "confidence": 0.9}, ...]}
```

## Findings

1. **The main inference behaviours of Jev can be reproduced on an ordinary open decoder LLM.** Direct readout, a shared state,
   sibling isolation (leakage into sibling questions 0, negative control 0.995) and listwise option interaction (a fifth option
   moves the log-odds) were all confirmed at 1.7B / 14B / 32B.
2. **The speed-up from shared computation is real; "not generating" by itself is almost no speed-up.** generate ≈ naive
   (0.9-1.1x), while for a long state and many questions (S=8k, Q=100) packed / shared run 53-73x faster than naive (on MPS).
3. **Within the range tested, decision accuracy is set mainly by backbone capability.** 1.7B → 14B → 32B gives MMLU
   0.554 → 0.750 → 0.809 and JevBench hard 0.423 → 0.550 → 0.622. LoRA on 4,800 examples barely helps at 14B and above, and
   the gap to Jev stays at roughly 10-12 points on both MMLU and JevBench hard.
4. **Much of the calibration "magic" can be reproduced with a scalar correction, but domain shift is unsolved.** The 32B reaches
   ECE 0.023 on MMLU (Jev's reported value: 0.031), and the same temperature transfers to JMMLU, transfers partially to JevBench
   hard (0.274 → 0.107) and is harmful on bridge: the best scale depends on the task.
5. **External measurement reproduced the public numbers, and the gap to Jev did not shrink.** Benchmark Heaven measured the 32B
   zero-shot configuration first through a tunnel to our machine (JevBench v1.2.7, partial row: easy 1.000 / standard 0.958 /
   judge 0.925 / public hard 0.622, Score 67.2) and then, from the published serving code on their own H100, over all 534
   decisions (v1.2.8): **#8 of 36, JevBench Score 70.1**, hard 0.645 on all 220 items (Jev 0.741), Calibration 79.0.

## What was built

With one model and one prompt, five inference structures (generate / naive / kvcache / packed / shared) and two orthogonal
readouts (full-vocabulary projection / option rows only) can be switched and compared.

| engine | stage | structure | purpose |
|---|---|---|---|
| `generate` | A | ordinary `generate`, emit a letter and parse it | baseline (the generating case) |
| `naive` | B | full forward of prefix+suffix per question, read only the `A/B/C...` logits at the last position | the effect of "not generating" alone |
| `readout="rows"` | B' | no full-vocabulary projection; compute the logits from the option-letter rows of the LM head only (available in every engine) | proof that dropping the vocab projection changes nothing |
| `kvcache` | D1 | prefill the state once → replicate the KV cache and batch the questions | shared computation (HF cache style) |
| `packed` | D2 | `[state \| q1 \| q2 \| ...]` as one sequence; a block attention mask lets each question see the state and itself only | one block/tree-attention implementation consistent with Hume's observations (reference implementation) |
| `shared` | D3 | the same packed input computed by a custom attention without a mask: the branch queries attend to the shared prefix together (Hydragen-style decomposition), each branch runs a small causal attention over itself, and the two are combined by log-sum-exp | an implementation that actually exploits block sparsity in compute and memory (no L×L mask) |

The `packed` attention mask and position ids:

```
            state     q1    q2    q3
state       ◤causal
q1          █████     ◤
q2          █████           ◤
q3          █████                 ◤        position_ids: state 0..S-1, each q restarts at S
```

In fp32 the choice logits of `naive` / `kvcache` / `packed` agree within 1e-4 (`tests/test_engines_equivalence.py`).
So D2 performs numerically the same computation as "Q independent prefix+q_i forwards", holding one copy of the prefix K/V
and running one forward.

The readout takes the single tokens `" A"`, `" B"`, ... out of the vocabulary logits and applies a softmax to them (it does
not read label names). `readout="rows"` (B') computes the same logits directly as `h @ W[choice_ids].T`; in fp32 it matches B
within 1e-4 (`tests/test_engines_equivalence.py`), which shows that the 150k-dimensional projection is not needed for the
decision. `confidence` is the post-hoc index `(p_max − 1/K) / (1 − 1/K)` that Hume confirmed in the official TypeSafe adapter;
it is not a probability (0 for a uniform distribution, 1 for one-hot). The entropy version `1 − H(p)/log K` is returned
separately as `entropy_concentration`.

**Qwen3's thinking mode is disabled and pinned.** Qwen3's chat template defaults to `enable_thinking=True`, so reading the
logits right after the assistant turn would compare A/B/C at a position where the model actually wants to start `<think>`.
jqv pins a non-thinking prompt in `jqv/prompt.py` that starts the assistant side with `<think>\n\n</think>\n\n` + `Answer:`,
and `tests/test_prompt.py` verifies that it matches the official `apply_chat_template(enable_thinking=False)` exactly.
generate (A) uses the same prefix + suffix, so the A/B/D1/D2 comparison is not confounded by reasoning.

**packed is not a reproduction of Jev; it is a reference implementation showing that the behaviours observed in Jev
(shared state, sibling isolation, direct probability readout) can be reproduced on an open Qwen.** Hume himself notes that
sibling isolation could be achieved by mechanisms other than a tree mask and that the exact attention mask is unknown.

## Setup

```bash
uv sync                        # Python 3.12, torch (MPS), transformers 5.x
uv run pytest                  # downloads Qwen/Qwen3-1.7B on first run
```

Runs on macOS / Apple Silicon (MPS) with HF Transformers only; a CUDA environment works unchanged.
The model is selected with `JQV_MODEL` (default `Qwen/Qwen3-1.7B`) and the dtype with `JQV_DTYPE` (default bf16).
A `*-Base` model automatically switches to a plain prompt without the chat template.

## Usage

```bash
# API server (the temperature file reuses T=11.9 fitted on English MMLU; not validated on the bridge-inspection domain.
# The response's calibration.dataset records where it came from.)
uv run python -m jqv.server --engine packed --temperature-file results/mmlu_packed_qwen3-1.7b_temperature.json
curl -s localhost:8000/decision -H 'content-type: application/json' -d @- <<'JSON'
{"state": "Bridge A: extensive corrosion on the lower flange of the main girder. No cracking in the deck. Bearings in good condition.",
 "questions": [{"question": "What damage does the main girder show?", "choices": ["none", "corrosion", "cracking"]},
               {"question": "Is there cracking in the deck?", "choices": ["yes", "no"]}]}
JSON

# Python
from jqv.model import load_runtime
from jqv.engine import make_engine
from jqv.types import Question
rt = load_runtime()
eng = make_engine("packed", rt, temperature=12.9)
eng.decide(state, [Question(question="...", choices=["...", "..."])])
```

## Experiment scripts

```bash
uv run scripts/eval.py --dataset mmlu  --engine packed --n 1200 --n-val 400   # accuracy / NLL / Brier / ECE
uv run scripts/eval.py --dataset jmmlu --engine packed --n 1200 --n-val 400
uv run scripts/fit_temperature.py results/mmlu_packed_qwen3-1.7b.npz          # fit T on val, before/after on test + reliability diagram
uv run scripts/bench.py --state-tokens 500 2000 8000 --questions 1 10 100     # latency / questions per second per engine
uv run scripts/isolation_test.py                                              # Hume's secret-code experiment and the fifth-option experiment
```

- `eval.py` groups questions that share a state into one `decide()` call (MMLU/JMMLU have an empty state, so N questions go in one batch).
- `fit_temperature.py` fits temperature scaling on the val split and reports ECE / Brier / NLL before and after on the test split.
  T minimises the NLL over a log grid on [0.05, 100] followed by a golden-section search (so it does not diverge on small,
  nearly separable data).
- `transfer_temperature.py` uses only the npz caches and reports ECE / NLL / Brier for T transferred across every source × target pair.
- `isolation_test.py` uses `packed_causal` (naive concatenation without the block mask) as a negative control to show that
  sibling-question information does not leak.
- `permutation_test.py` measures probability changes under `--mode label` (rotate the letters only), `--mode order` (rotate the
  option order only) and `--mode fifth` (add an unrelated fifth option). `Question.labels` sets the letter at each position (experiments only).
- `bench.py` appends each condition's result to `results/bench_<model>.jsonl` as it completes and prints an estimate of the
  remaining time; the same command resumes after an interruption.

Results are written to `results/` as JSON / PNG / Markdown.

### Where a calibration temperature comes from (provenance)

A temperature is only valid for the distribution it was fitted on. The temperature file written by `fit_temperature.py`
contains `model`, `engine`, `prompt_hash` (derived from the prompt format and system prompt), `dataset`, `n_val`,
`choice_counts` and `fitted_at`, and the server checks `model` and `prompt_hash` against the runtime at start-up. On a
mismatch it refuses to start (before loading the model); only with `JQV_ALLOW_CALIBRATION_MISMATCH=1` does it continue with a
warning. Old-format files (`temperature` only) still load but are treated as unverified. The `/decision` response carries
`calibration` (the temperature and the metadata above), or `null` when no temperature is set; `/health` reports the same
information together with the `prompt_hash`.

## Results (Qwen3-1.7B, bf16, M5 Max)

### Accuracy and calibration (`packed` engine, val 400 / test 800, T fitted on val)

| dataset | accuracy | ECE before | ECE after | Brier before | Brier after | NLL before | NLL after | T |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| MMLU (en)  | 0.554 | 0.413 | **0.080** | 0.852 | 0.576 | 5.79 | 1.08 | 12.0 |
| JMMLU (ja) | 0.466 | 0.485 | **0.066** | 0.983 | 0.639 | 6.13 | 1.19 | 12.8 |
| bridge_synth (30 items, uncalibrated) | 0.933 | 0.051 | - | 0.079 | - | 0.10 | - | - |

The raw 1-token readout is extremely over-confident (mean confidence 0.96), but a single scalar temperature brings the ECE
down to 0.06-0.08 (still short of the ECE 0.031 that Hume measured for Jev). Reliability diagrams are in
`results/*_reliability_{before,after}.png`.

### Temperature transfer (`scripts/transfer_temperature.py`, `results/transfer_qwen3-1.7b.json`)

A T fitted on the val split of one source was applied unchanged to the test split of another target. Against Jev's ECE
0.031 (Hume's measurement on a 1,200-item MMLU sample; the prompt conditions cannot be confirmed from the article), the
off-diagonal cells are the closer comparison, not the in-distribution diagonal.

| T fitted on → applied to | T | MMLU test (n=800) ECE | JMMLU test (n=800) ECE | bridge (n=30) ECE |
|---|---:|---:|---:|---:|
| none (T=1) | 1.00 | 0.413 | 0.485 | **0.051** |
| MMLU val | 11.96 | **0.080** | 0.082 | 0.237 |
| JMMLU val | 12.84 | 0.081 | **0.066** | 0.266 |
| MMLU+JMMLU val | 12.34 | 0.085 | 0.072 | 0.253 |
| oracle (fitted on each target's own test) | 11.5 / 12.9 / 2.94 | 0.092 | 0.070 | 0.047 |

- **Transfer across languages holds.** T=12.0 fitted on MMLU gives ECE 0.082 on JMMLU (in-distribution 0.066); the reverse
  direction gives 0.081 (0.080). For English and Japanese knowledge questions the distortion of the raw logit scale is
  almost the same (T ≈ 12).
- **Transfer across task types does not hold.** On bridge (a reading-type task whose answer is in the state, accuracy 0.93)
  the raw probabilities are already nearly calibrated (ECE 0.051, oracle T=2.9); applying T=12 drops the mean confidence from
  0.98 to 0.73, making the model under-confident, and the ECE worsens to 0.24.
- So the distortion of Qwen3-1.7B's 1-token logits is not a constant scale: about 12x on closed-book knowledge questions and
  about 3x on questions answerable from the state. One post-hoc scalar does not calibrate a general Decision API, and this
  is where the difference from Jev (a distribution trained with RLCD) lies. bridge has 30 synthetic items, so its numbers are
  indicative only. The transfer criterion used: an off-diagonal ECE within the diagonal + 0.02.

### The prior of the letter labels (`scripts/permutation_test.py --mode label`, `results/permutation_label_qwen3-1.7b.json`)

Keep the position and content of the options fixed and rotate only the letters in front of them
(`A. x / B. y / C. z` → `B. x / C. y / A. z` → ...). Results are aggregated back onto the semantic options, so the only
remaining difference is which letter token is read. Measured on raw probabilities (T=1).

| dataset | mean probability per letter A / B / C / D (0.25 if uniform) | mean abs. change of p(correct) across rotations | argmax identical across all rotations | accuracy range |
|---|---|---:|---:|---|
| MMLU (n=300) | 0.31 / 0.28 / 0.22 / 0.19 | 0.174 | 54% | 0.533-0.560 |
| bridge (n=30) | 0.30 / 0.31 / 0.24 / 0.24 | 0.039 | 93% | 0.900-0.955 |

- **The prior of the letter tokens is not negligible.** On MMLU, A/B are systematically higher than C/D (0.31 vs 0.19), and
  relabelling alone changes the argmax on 46% of the items. Mean accuracy hardly moves (0.53-0.56), so the prior tips the
  undecided items towards A/B.
- On bridge, where the answer is in the state, the effect is small (93% identical). Strong evidence overrides the prior.
- Caveat: rotating the letters produces unnatural prompts with letters out of order (`B. C. D. A.`). The design isolates the
  effect of the letter at the cost of including that unnaturalness. The effect of the order itself is measured in the order
  experiment below.

### Option order and a fifth option (`--mode order`, `--mode fifth`)

order: keep the letters fixed as A, B, C, D and rotate only the option texts (the listwise position effect). Same metrics as
the label experiment.

| dataset / mode | mean probability per slot (label: letters A/B/C/D; order: positions 1/2/3/4) | mean abs. change of p(correct) | argmax identical | accuracy range |
|---|---|---:|---:|---|
| MMLU / label (n=300) | 0.31 / 0.28 / 0.22 / 0.19 | 0.174 | 54% | 0.533-0.560 |
| MMLU / order (n=300) | 0.25 / 0.31 / 0.22 / 0.22 | **0.220** | **48%** | 0.547-0.580 |
| bridge / label (n=30) | 0.30 / 0.31 / 0.24 / 0.24 | 0.039 | 93% | 0.900-0.955 |
| bridge / order (n=30) | 0.29 / 0.32 / 0.24 / 0.23 | 0.084 | 87% | 0.909-0.967 |

- **The order effect is larger than the letter prior.** Changing the order on MMLU changes the argmax on 52% of the items and
  moves p(correct) by 0.22 on average. Position 2 is the most likely to be chosen (0.31). The letter prior (towards A/B) and
  the position prior (towards the second slot) exist independently.
- In both experiments mean accuracy hardly moves, so these priors act as tie-breakers on weak-evidence items. Because the
  uncalibrated probabilities are close to one-hot, flipping the tie-break swaps p(correct) between 0 and 1, which is why the
  mean absolute change is large.
- Hume reports that Jev's probabilities also move with option order; on a vocabulary-readout Qwen the sensitivity is
  considerably larger. Averaging over the order (returning the mean over all cyclic rotations) is a practical fix at K times
  the cost; the principled fix is training a pointer head with order shuffling.

fifth: add an unrelated fifth option "none of the above / unknown" to the 4 choices and measure the change in the log-odds of
the correct answer against the strongest wrong answer (Hume's Jev: −0.28, 95% CI −0.36 to −0.19).

| dataset | n | mean Δlog-odds ± 95% CI | per-item SD | mean probability on the fifth option | share of items whose argmax among the original 4 changes |
|---|---:|---:|---:|---:|---:|
| MMLU | 300 | −0.04 ± 0.45 | 3.96 | 0.19 | 6.7% |
| bridge | 21 | −0.49 ± 0.39 | 0.92 | ≈0 | 0% |

- On MMLU **no systematic shift is detected** (the CI straddles 0), but individual items move by as much as ±4 nat (SD 3.96)
  and the argmax among the 4 changes on 6.7%. So the option list influences the final hidden state jointly (this is not
  independent logits + softmax), but not in the consistent contracting direction seen in Jev; it swings strongly per item.
- On bridge the direction matches Jev (−0.49) but with n=21 the CI barely excludes 0.
- The 19% of probability landing on the fifth option on MMLU is partly because a Japanese "none of the above / unknown" was
  added to English knowledge questions; a re-measurement with the option wording in the same language is needed.

### Isolation (reproducing Hume's secret-code experiment)

| engine | p(secret) alone | secret in a sibling question | secret in the state |
|---|---:|---:|---:|
| naive / kvcache / packed | 0.000 | **0.000** | 1.000 |
| packed_causal (negative control: naive concatenation) | 0.000 | **0.996** | 1.000 |

With the block mask, `packed` sees nothing of the sibling questions (the same behaviour as Jev). Remove the mask and it leaks completely.

For the fifth-option experiment see the "Option order and a fifth option" section above (n=300); the n=21 version inside
`isolation_test.py` is an older measurement.

### Throughput

Qwen3-1.7B / bf16 / Apple M5 Max. Per condition: one warm-up + the median of 3 timed runs (1 run where the warm-up exceeded
60 s). A question is about 55 tokens. All 36 conditions are in `results/bench_qwen3-1.7b.md`.

| state tok | Q | generate (A) | naive (B) | kvcache (D1) | packed (D2) | shared (D3) | best vs B |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 538  | 10  | 0.61 s | 0.53 s | 0.14 s | **0.12 s** | 0.13 s | 4.5x |
| 538  | 100 | 7.2 s  | 6.8 s  | 1.08 s | 1.20 s | **0.95 s** | 7.2x |
| 2038 | 10  | 3.1 s  | 3.3 s  | 0.44 s | **0.40 s** | 0.43 s | 8.2x |
| 2038 | 100 | 43 s   | 41 s   | 2.3 s  | 2.1 s  | **1.8 s** | 23x |
| 2038 | 1000 | -     | -      | -      | **9.9 s** | 13.3 s | - |
| 8038 | 10  | 23 s   | 23 s   | 2.1 s | 2.4 s | **1.6 s** | 14x |
| 8038 | 100 | 231 s  | 256 s  | 6.6 s  | 4.8 s  | **3.5 s** | **73x** |
| 8038 | 1000 | -     | -      | 64 s   | **15.9 s** | 22.5 s | - |

Q=1000 was not measured for naive / generate (over 40 minutes). packed at Q=1000 runs in chunk mode: the prefix goes into the
KV cache and the questions are packed 2048 tokens at a time (`chunk_tokens=2048`; 16384 takes 36 s, and smaller tiles waste
less on the masked branch-branch blocks).

At Q=1 all four engines are equivalent (there is nothing to share; at S=8038 both naive and packed take 0.79 s).

B vs B' (readout full / rows, packed, Q=100, medians of 5 alternating measurements in one process, `results/readout_ab_qwen3-1.7b.json`):

| state tok | full (B) | rows (B') | rows / full |
|---:|---:|---:|---:|
| 538  | 1.19 s | 1.14 s | 0.96 |
| 2038 | 1.62 s | 1.56 s | 0.96 |

The difference is about 4% (40-60 ms at Q=100, the (100 × 151k) LM-head projection plus the softmax), negligible next to the
backbone. The value of B' is not speed but the separation it proves: the decision is identical without the vocabulary projection.
The `:rows` rows in `results/bench_qwen3-1.7b.md` were measured in a separate process and include run-to-run variation (±30%).

What this shows:

1. **"Not generating" alone is not faster.** generate and naive are the same at 0.9-1.1x. Prefill dominates; decoding 4 tokens
   is noise. The value of B is not speed but that a probability distribution comes out and can be calibrated.
2. **The benefit of shared computation grows with state length × number of questions.** At S=8k, Q=100, packed is 53x naive
   and kvcache 39x. This has the same shape as Hume's observation for Jev: the state length dominates and the marginal cost of
   a question is almost zero.
3. **packed beats kvcache more the longer the state.** kvcache replicates the HF cache per batch row, so at an 8k state the
   batch drops to 8 under memory pressure; packed holds one copy of the prefix K/V. For short states kvcache is marginally
   faster because of the mask-construction overhead.
5. **shared (D3) is 1.15-1.5x faster than packed as long as prefix + all questions fit in one forward (≤ 16k tokens), and loses
   to packed's small tiles in the Q=1000 chunk regime (see the "D3" section below).**
4. In absolute terms this is 1-2 orders of magnitude slower than Jev (about 160 ms for 30k tokens); that is the environment
   (a 1.7B on MPS) and does not affect the structural comparison.

## D3: shared-prefix attention without a mask (`jqv/engine/shared.py`)

packed (D2) passes a `(1, 1, L, L)` mask to SDPA, so the attention is computed densely. D3 takes the same packed input
(`[prefix | q1 | ... | qQ]`, position ids restarting right after the prefix for each branch) and computes it with a custom
attention plugged in through `AttentionInterface.register` of Transformers 5.x.

```
prefix rows : ordinary causal attention (fused SDPA, no mask)
branch rows : (a) all branch queries attend together to the shared prefix keys ... one (Σq) × S block, no mask
              (b) each branch attends causally to its own keys ... branches padded and batched, (Q, qmax, qmax)
              (a) and (b) are combined by log-sum-exp (the same identity flash attention uses)
```

The attention cost goes from O(L²) to O(S²/2 + (Σq)·S + Σq²); the largest temporary is a (rows × S) block, and neither L×L
nor Lc×(S+Lc) is ever materialised. In fp32 it matches naive / packed within 1e-3 and passes the isolation test.

Computing (a) with fused SDPA needs the partition function Z = Σ exp(s) for the combination, but SDPA returns only the
normalised output. So **one zero key with score 0 is appended, its value set to a probe (1 in channel 0 only), and SDPA is
called a second time.** The probe channel's output is c = 1/(Z + P) (P = number of padded zero keys), giving Z = 1/c − P
(Qwen3 has q/k RMSNorm, so |s| < 50 or so and c does not underflow in fp32 or bf16). `backend="manual"` (matmul + softmax
statistics in chunks) is kept as well; on MPS it is memory-bound and slower than fused (57 s vs 49 s at S=8k, Q=1000 when measured).

**Measured on MPS (table above)**: within one forward, shared is faster than packed (S=8038, Q=100: 3.5 s vs 4.8 s, 73x vs
naive). In the Q=1000 chunk regime packed (2048-token tiles) is faster (15.9 s vs 22.5 s), for two reasons.

- SDPA on MPS does not expose the partition function, so (a) needs **two full passes** (the probe call costs the same as the
  main one). Widening the head dim to 129/136 to add a probe channel is 10x slower on MPS (kernels other than `head_dim=128`
  are slow), so it cannot be used.
- With packed's small tiles the masked kernel wastes only Lc/(S+Lc) ≈ 20%, and one fused kernel call suffices even with the mask.

So on MPS the number of fused kernel calls matters more than physically exploiting sparsity. On CUDA, FlexAttention's
`BlockMask` (`backend="flex"`, not verified in this environment) can do (a) and (b) in one block-sparse kernel, where this
structure should deliver its real performance. At Jev's scale (a 23k-token state × 5,000 questions) packed's L×L equivalent
(Lc×(S+Lc) even with chunks) is not viable, and a D3-type decomposition becomes mandatory.

## C: a trained decision head (`jqv/heads.py`, `jqv/train/`, `scripts/train_head.py`)

Instead of the vocabulary readout (B), a head is trained that emits decision logits directly from the final hidden state.
The prompt is identical to B.

| head | formula | properties |
|---|---|---|
| slot (C1) | `z = W h_d + b` (h_d is the hidden state at the `Answer:` position; rows = option slots) | Hume's "slot head". Initialised from the letter rows of the LM head, step 0 is identical to B' |
| pointer (C2) | `z_i = (U h_d)·(V h_i)/√r + w·h_i` (h_i is the hidden state at the end of option i's text) | variable K, equivariant to order (`tests/test_heads.py`), arbitrary labels. A candidate for Hume's "pointer scorer" |

- Training data: MMLU `auxiliary_train` (99,842 items). The option order is shuffled every time so that no position prior is
  learned. Validation: 256 items from MMLU `validation`. Evaluation: the same MMLU / JMMLU test 800 as everywhere else.
- The LLM is frozen; a setting with LoRA (r=16, q/k/v/o_proj) is compared with a head-only setting without LoRA.
- Training runs on MPS with `python -u scripts/train_head.py`: loss / step/s / ETA every 10 steps, a checkpoint
  (`results/train/<run>/last`) and validation every 100 steps (best in `best/`), and `--resume` continues from a checkpoint.
- Inference: `make_engine("pointer", rt, head_dir="results/train/<run>/best")`. The head records the model and prompt_hash
  it was trained with and refuses a mismatch.

### Results (600 steps × batch 8 = 4,800 examples, 11-24 min on MPS, test 800. raw = uncalibrated, +T = temperature fitted on val 400)

| method | trained params | MMLU acc | MMLU NLL raw / +T | MMLU ECE raw / +T | JMMLU acc | JMMLU NLL raw / +T | JMMLU ECE raw / +T |
|---|---:|---:|---:|---:|---:|---:|---:|
| B: vocabulary readout (no training) | 0 | 0.554 | 5.79 / 1.08 | 0.41 / 0.080 | 0.466 | 6.13 / 1.19 | 0.49 / 0.066 |
| C1: slot + LoRA r=16 | 6.5M | **0.575** | 1.06 / **0.98** | 0.14 / 0.044 | **0.505** | 1.23 / **1.12** | 0.16 / **0.029** |
| C2: pointer + LoRA r=16 | 7.5M | 0.561 | 1.10 / 1.05 | **0.11** / 0.048 | 0.475 | 1.38 / 1.24 | 0.15 / 0.039 |
| C2: pointer, LLM frozen | 1.0M | 0.471 | 2.00 / 1.19 | 0.32 / 0.028 | 0.395 | 2.34 / 1.32 | 0.35 / 0.029 |

Order sensitivity (MMLU 300 items × 4 cyclic rotations, `--mode order`):

| method | mean probability per position 1/2/3/4 | mean abs. change of p(correct) | argmax identical |
|---|---|---:|---:|
| B | 0.25 / 0.31 / 0.22 / 0.22 | 0.220 | 48% |
| C1 slot + LoRA | 0.26 / 0.26 / 0.25 / 0.24 | 0.126 | 55% |
| C2 pointer + LoRA | 0.20 / 0.25 / 0.32 / 0.23 | 0.164 | 48% |

What this shows:

1. **At 1.7B, decision training greatly improves raw calibration, and accuracy improves significantly on JMMLU (the +2.1 points
   on MMLU is not significant).** Heads with LoRA are more accurate than B (MMLU +2.1, p=0.125 / JMMLU +3.9, p=0.006; see
   "Reading the accuracy numbers"); at 14B and above the accuracy gain disappears (backbone scaling section). Above all, the
   raw probabilities are usable from the start (ECE 0.41 → 0.11-0.14, NLL 5.8 → 1.1). With a temperature they reach ECE
   0.03-0.05 and NLL 0.98, better than B + temperature (0.080, 1.08). The JMMLU ECE of 0.029 is at the level of Jev's MMLU
   0.031, but here it includes an in-distribution temperature.
2. **Training a new head on a frozen LLM is worse than B** (MMLU 0.471, −8 points). The LLM has not learned where in the
   hidden state "option i is correct" lives, so a head alone cannot read it out; the representation has to be moved with LoRA.
   A temperature fixes the ECE but not the accuracy.
3. **slot (initialised from the LM-head letter rows) ≥ pointer (trained from scratch).** At this amount of training, slot's
   head start from B' is an advantage. pointer is order-equivariant on paper, but h_i itself depends on the earlier options
   through causal attention, so an order effect remains (argmax identical 48%). slot trained with order shuffling has an
   almost uniform position prior (0.26/0.26/0.25/0.24) and a higher identical-argmax rate (48% → 55%).
4. A head trained on English MMLU transfers to Japanese JMMLU (accuracy and NLL both improve). On bridge (Japanese, reading
   type, 30 items) pointer + LoRA scores 0.833, below B's 0.933; task types outside the training distribution need care.
5. Training leaves per-step loss and validation values in `results/train/<run>/train_log.jsonl` and checkpoints in `best/` and
   `last/`. pointer + LoRA's best is step 300 (val NLL 1.145); after that it overfits somewhat.

### E: calibration-oriented training `L = CE + λ·Brier` (slot + LoRA, λ ∈ {0, 0.5, 1, 2}, everything else identical)

| λ | MMLU acc | NLL raw / +T | Brier raw / +T | ECE raw / +T | T | JMMLU acc | NLL raw / +T | ECE raw / +T | bridge NLL / ECE (T=1) |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0   | 0.575 | 1.056 / **0.977** | 0.554 / **0.523** | 0.137 / **0.044** | 1.82 | 0.505 | 1.233 / 1.118 | 0.158 / 0.029 | 0.064 / 0.056 |
| 0.5 | 0.573 | 1.052 / 0.979 | 0.551 / 0.524 | 0.135 / 0.051 | 1.78 | 0.505 | 1.220 / 1.119 | 0.150 / 0.036 | 0.057 / 0.050 |
| 1   | **0.579** | **1.049** / 0.981 | 0.552 / 0.524 | 0.128 / 0.051 | 1.76 | **0.507** | 1.217 / 1.119 | 0.147 / **0.028** | 0.061 / 0.053 |
| 2   | 0.549 | 1.069 / 1.043 | 0.571 / 0.560 | **0.082** / 0.046 | 1.47 | 0.480 | 1.217 / 1.171 | **0.098** / 0.044 | 0.147 / 0.120 |

- **For λ ≤ 1 the results are indistinguishable from CE alone.** The differences in accuracy / NLL / ECE are inside the
  confidence interval for 800 items (±3.4 points, about ±0.01 ECE).
- **λ=2 flattens the raw probabilities** (mean confidence 0.71 → 0.63, raw ECE 0.137 → 0.082) but costs 3 points of accuracy,
  and after a temperature the NLL / Brier are worse than λ=0 (0.977 → 1.043). The Brier term does not add discrimination; it
  builds a temperature into training. That has value when no calibration set is available, but if one temperature can be
  fitted on val, CE + T is better.
- **Transfer across task types is not solved by any λ.** On bridge T=1 is best for every λ, and applying the MMLU T worsens
  the ECE to 0.15-0.19. λ=2's raw bridge ECE (0.120) is worse than λ=0's (0.056).
- Against Jev's ECE 0.031 (Hume's measurement, prompt conditions unknown), slot + LoRA + an in-distribution T gives 0.028-0.029
  on JMMLU and 0.044-0.051 on MMLU. "Training + temperature" reaches the same level numerically, but Jev's RLCD produces
  0.03 without a temperature and across distributions, which is a different condition.
- At 1.7B, λ=1 had the lowest MMLU val NLL (1.089 vs 1.103 for λ=0), a difference within noise. After the 14B re-sweep
  **λ=0** was adopted, and the 32B also uses λ=0 (backbone scaling section).

## Training-free accuracy gains: a few-shot state and cyclic-rotation averaging

The goal is about 80% on MMLU. The main lever is the backbone (14B / 32B, below), but training-free measures were settled
on the 1.7B first: (a) **few-shot examples in the shared state** (with packed the prefill happens once per subject, so the
per-question cost does not grow), (b) **cyclic-rotation averaging `perm_avg`** (rotate the option order K ways, map back to
the semantic options and average the probabilities; cancels the position and letter priors).
`scripts/eval.py --shots 5 [--shots-mode fixed] --perm-avg`, `make_engine(..., perm_avg=True)`.

| setting (1.7B, packed, accuracy at T=1) | MMLU acc | Δ vs zero-shot [95% CI] | p | ECE raw / +T | JMMLU acc | Δ | p | ECE raw / +T |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| zero-shot | 0.554 | - | - | 0.413 / 0.080 | 0.466 | - | - | 0.485 / 0.066 |
| 5-shot, same subject (MMLU dev) | 0.550 | −0.004 [−0.036, +0.030] | 0.88 | 0.413 / 0.103 | 0.469 | +0.003 | 0.94 | 0.494 / 0.048 |
| 5-shot, fixed (the same 5 examples for every question) | 0.422 | **−0.131** [−0.169, −0.094] | <0.001 | 0.546 / 0.047 | 0.421 | −0.045 | 0.014 | 0.529 / 0.061 |
| perm_avg (K=4) | **0.584** | **+0.030** [+0.001, +0.058] | 0.050 | **0.195** / **0.063** | 0.485 | +0.019 | 0.18 | 0.291 / 0.034 |
| 5-shot same subject + perm_avg | 0.578 | +0.024 [−0.009, +0.058] | 0.17 | 0.214 / 0.075 | **0.497** | +0.031 | 0.077 | 0.282 / 0.057 |

- **Cyclic-rotation averaging gives +3 points without training** (MMLU, p=0.05). The raw ECE also drops from 0.41 to 0.20 (mean
  confidence 0.96 → 0.78) and the temperature moves closer to 1 (12 → 8.8). The gain comes from averaging out the instability
  seen in the order experiment (52% of argmaxes change with the order); it matches the accuracy of the trained slot + LoRA
  (0.575-0.579) at inference time only. Selective accuracy after calibration also improves (at p ≥ 0.7, 27% of the items at
  accuracy 0.89; zero-shot: 19% at 0.87).
- **The cost is 1.7x** (packed S=2038, Q=100: 48.6 → 28.1 q/s). The branch tokens are 4x but the state prefill is shared, so
  it is not 4x.
- **Same-subject 5-shot does not help** (±0.4 points). The direct readout of an instruct model already understands the
  format, and examples add no information. On JMMLU (with English examples) the effect is likewise nil.
- **Five fixed examples shared by every question hurt badly** (MMLU −13 points). Predicted letters concentrate on B at 64%
  (zero-shot 32%); unrelated examples steer the letter prior. If examples are shared in a Jev-style "one state, many
  questions" setting, the content and the answer-letter bias of the examples can distort the decisions.
- Settings carried to 14B / 32B: **zero-shot and perm_avg**. Few-shot is checked once in its same-subject version only.

### Reading the accuracy numbers (`scripts/compare_runs.py`)

Accuracy is the argmax hit rate and does not change with the temperature. For a Decision API the questions are whether the
probabilities are honest (ECE / NLL / Brier) and what accuracy and coverage you get when only high-confidence items are
processed automatically. The 95% confidence interval for test 800 is ±3.4 points, so smaller differences are judged with a
paired comparison on the same 800 items (McNemar).

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
| B (vocab) | 67% at 0.64 | 19% at 0.87 | 0% |
| slot+LoRA | 54% at 0.75 | 29% at 0.89 | 7% at 1.00 |
| pointer+LoRA | 52% at 0.74 | 24% at 0.83 | 3% at 0.88 |
| pointer frozen | 33% at 0.65 | 7% at 0.89 | 1% at 1.00 |

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
| B (vocab) | 52% at 0.59 | 5% at 0.87 | 0% |
| slot+LoRA | 45% at 0.68 | 18% at 0.81 | 3% at 0.96 |
| pointer+LoRA | 32% at 0.63 | 8% at 0.70 | 0% at 0.00 |
| pointer frozen | 11% at 0.64 | 1% at 1.00 | 0% |

- "slot + LoRA beats B" is significant on JMMLU (p=0.006) and not on MMLU (p=0.125). "pointer + LoRA equals B" and "the frozen
  head is worse" are solid.
- Even with only 2 points of overall accuracy between them, after calibration slot + LoRA can process 1.5x as many items as B
  (29% vs 19%) at the same accuracy 0.89 when cutting at p ≥ 0.7. For bulk classification where only the uncertain items go to
  a person, this selective accuracy is what matters.
- MMLU 0.55 here is a zero-shot, non-thinking, 1,200-item-subset number and is not comparable with the official evaluation
  (5-shot, all 14,042 items). bridge has 30 items (one item = 3.3 points), so it is a functional check only.

## Backbone scaling (`scripts/scaling_table.py`, `results/scaling_table.md`)

Same code, prompt and 1,200-item subset (seed 0, val 400 / test 800); only the backbone changes. Restricted to dense Qwen3:
MoE (30B-A3B) and Qwen3.5 (hybrid attention) are excluded because they change the attention / KV discussion. The Jev row is
Hume's measurement (a 1,200-item MMLU sample, the probabilities as returned by the API; the prompt conditions cannot be
confirmed from the article).

| backbone | params | B zero-shot: MMLU / JMMLU | B ECE raw / +T (T) | B + perm_avg: MMLU / JMMLU | perm_avg ECE raw / +T | 5-shot + perm_avg: MMLU | slot+LoRA: MMLU / JMMLU | slot ECE raw / +T | slot + perm_avg: MMLU | JevBench hard (public 111): B / perm_avg / slot | packed q/s (S=2k, Q=100) plain / perm_avg |
|---|---:|---:|---|---:|---|---:|---:|---|---:|---:|---:|
| qwen3-1.7b | 1.7B | 0.554 / 0.466 | 0.413 / 0.080 (12.0) | 0.584 / 0.485 | 0.195 / 0.063 | 0.578 | 0.575 / 0.505 (slot_lora) | 0.137 / 0.044 | - | 0.423 / 0.432 / 0.414 | 48.6 / 28.1 |
| qwen3-14b | 14.8B | 0.750 / 0.710 | 0.207 / 0.042 (5.1) | 0.781 / 0.729 | 0.113 / 0.047 | 0.782 | 0.757 / 0.711 (qwen3-14b_slot_brier0) | 0.114 / 0.045 | - | 0.550 / 0.568 / 0.586 | 9.3 / 4.4 |
| qwen3-32b | 32.8B | 0.809 / 0.771 | 0.137 / 0.023 (3.0) | 0.812 / 0.790 | 0.087 / 0.034 | - | 0.801 / 0.782 (qwen3-32b_slot_best) | 0.115 / 0.031 | 0.819 | 0.622 / 0.640 / 0.622 | 2.5 / - |
| Jev (TypeSafe, Hume 2025) | ? | **0.918** / - | 0.031 (Hume, 1,200 items, API probabilities as returned) | - | - | - | - | - | - | **0.741** (534 decisions, Benchmark Heaven) | 30k tok in ~160 ms |

### What 1.7B → 14B showed

- **Within this range, backbone scale was the dominant factor for accuracy.** Zero-shot direct readout: MMLU 0.554 → 0.750
  (+19.6), JMMLU 0.466 → 0.710 (+24.4). Cyclic-rotation averaging, effective at 1.7B, helps by the same margin at 14B (+3.1,
  p=0.001); 5-shot gives +1.9 at 14B (p=0.12), and both together 0.782 (p=0.008). 0.78 at 14B without training; the gap to
  Jev's 0.918 is 14 points.
- **Raw calibration also improves with the backbone.** Raw ECE 0.41 → 0.21, temperature 12 → 5.1. After the temperature the
  ECE is 0.042 (MMLU) / 0.046 (JMMLU), within 0.01-ish of Jev's 0.031 without any training. Raw ECE after rotation averaging: 0.113.
- **Temperature transfer has the same structure as at 1.7B.** MMLU ↔ JMMLU transfers (T ≈ 5.0 gives ECE 0.040-0.048). bridge
  has ECE 0.000 at T=1 (all 30 correct, confidence 0.99999); applying the MMLU T worsens it to 0.053, a smaller harm than the
  0.24 at 1.7B.
- **The benefit of shared computation does not depend on the backbone.** At S=2038, Q=100 packed is 22x naive (20x at 1.7B),
  and generate ≈ naive (1.04x) holds too. Absolute speed: packed 9.3 q/s (1/5.2 of the 1.7B's 48.6 q/s, gentler than the 8.7x
  parameter ratio). shared (D3) is 11% faster than packed at S=8038, Q=100 and slower at Q=1000 (the same trend as 1.7B).
- **Isolation and option interaction are the same.** Leakage into sibling questions 0.0008 for packed / shared, negative
  control 0.995. The fifth-option Δlog-odds on the 21 bridge items is −1.24 ± 0.42 (1.7B: −0.49 ± 0.39): the negative shift
  Hume saw in Jev appears more strongly at 14B.
- **The equivalence tests pass exactly in fp32; in bf16 the logit difference is exactly 1 ulp (0.5).** The test tolerances
  were made dtype-dependent.
- 14B bench time: naive at Q=100 takes 238 s (1.7B: 41 s). From here on naive / generate are limited to S ≤ 2000.

### 14B slot + LoRA λ sweep (same settings as 1.7B: r=16, 600 steps × batch 8, option shuffling)

| λ | val NLL (best) | MMLU acc | Δ vs zero-shot [95% CI] | p | NLL raw / +T | ECE raw / +T | JMMLU acc | NLL raw / +T | ECE raw / +T |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| zero-shot B | - | 0.750 | - | - | 1.799 / 0.651 | 0.207 / 0.042 | 0.710 | 1.829 / 0.719 | 0.243 / 0.046 |
| perm_avg (reference) | - | **0.781** | +0.031 [+0.014, +0.049] | **0.001** | - / - | 0.113 / 0.047 | **0.729** | - / - | 0.141 / - |
| 0   | **0.675** | 0.757 | +0.007 [−0.011, +0.025] | 0.50 | 0.730 / **0.642** | 0.114 / 0.045 | 0.711 | 0.813 / **0.714** | 0.131 / 0.044 |
| 0.5 | 0.692 | 0.759 | +0.009 | 0.39 | 0.737 / 0.647 | 0.110 / 0.036 | 0.703 | 0.821 / 0.720 | 0.138 / 0.047 |
| 1   | 0.693 | 0.755 | +0.005 | 0.69 | 0.744 / 0.653 | 0.114 / 0.035 | 0.710 | 0.829 / 0.724 | 0.134 / 0.050 |
| 2   | 0.720 | 0.760 | +0.010 | 0.35 | 0.772 / 0.656 | 0.121 / **0.031** | 0.708 | 0.849 / 0.723 | 0.141 / 0.057 |

- **At 14B, decision training on 4,800 examples does not raise accuracy.** All four levels give +0.5-1.0 points (p ≥ 0.35);
  the +2-4 points seen at 1.7B vanish. On the same backbone perm_avg gives +3.1 (p=0.001), so order averaging at inference
  beats training.
- **Raw calibration improves with training** (ECE 0.21 → 0.11-0.12, NLL 1.80 → 0.73-0.77), but after a temperature it equals
  zero-shot + T (NLL 0.651, ECE 0.042). λ=2's MMLU ECE+T of 0.031 equals Jev's 0.031, but on JMMLU it is the worst at 0.057.
  Differences between λ values are within noise.
- The accuracy drop at λ=2 seen at 1.7B does not occur at 14B (0.760).
- The setting carried to 32B is **λ=0** (lowest MMLU val NLL 0.675, `results/scaling_best_config.json`). The 1.7B choice was
  λ=1, but both differences are noise and plain CE is the simplest.
- Training took 97-119 min per run (0.09-0.10 step/s, `--grad-accum 2`); evaluation 9-12 min per dataset. `--resume` was
  verified on the λ=1 run (20 steps → 600 steps).

### 32B: final check (B, perm_avg, slot + LoRA λ=0, slot + LoRA + perm_avg)

| 32B (test 800) | MMLU acc | Δ vs zero-shot [95% CI] | p | NLL raw / +T | ECE raw / +T | T | JMMLU acc | ECE raw / +T |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| zero-shot B | 0.809 | - | - | 0.947 / 0.535 | 0.137 / **0.023** | 3.0 | 0.771 | 0.158 / 0.041 |
| B + perm_avg | 0.812 | +0.004 [−0.013, +0.020] | 0.76 | 0.740 / 0.528 | 0.087 / 0.034 | 2.4 | **0.790** | 0.078 / **0.018** |
| slot + LoRA λ=0 | 0.801 | −0.007 [−0.021, +0.006] | 0.38 | 0.674 / 0.550 | 0.115 / 0.031 | 1.7 | 0.782 | 0.109 / 0.040 |
| slot + LoRA + perm_avg | **0.819** | +0.010 [−0.007, +0.028] | 0.34 | 0.583 / 0.533 | 0.063 / 0.042 | 1.4 | - | - |

Training: 600 steps × batch 8 (micro-batch 2 × accumulation 4, gradient checkpointing) in 302 min (0.03 step/s); memory use
123-127 GB, within the 128 GB. Best is step 300 (val NLL 0.688). Evaluation with the slot engine takes 22-24 min per dataset,
60 min with perm_avg.

### Conclusions from 1.7B → 14B → 32B

1. **Compared with the 4,800-example decision training tried here, backbone scale was the dominant factor for accuracy.**
   Zero-shot direct readout: MMLU 0.554 → 0.750 (+19.6) → 0.809 (+5.9); the gap to Jev (0.918) went 36 → 17 → 11 points. The
   remaining ~11 points are presumably the capability difference of the backbone itself or the effect of Jev's large-scale
   post-training. The 80% target was reached by the 32B zero-shot. Inference-side tricks (perm_avg) give +3 points at 1.7B /
   14B and shrink to +0.4 at 32B.
2. **In- or near-distribution on MMLU-type data, a scalar temperature alone reaches the ECE Jev reports; that temperature does
   not transfer universally to arbitrary tasks.** ECE+T is 0.080 → 0.042 → 0.023 (Jev 0.031). The temperature falls
   monotonically, 12 → 5.1 → 3.0, and T=2.6 fitted on JMMLU still gives 0.031 on MMLU. But the same T is only partly effective
   on JevBench hard (0.274 → 0.107) and harmful on bridge. Note that Jev's value is without a temperature, while jqv's raw ECE
   is 0.137 even at 32B (0.087 after perm_avg).
3. **Decision training on 4,800 examples does not raise accuracy at 14B and above.** slot + LoRA gives +2-4 points at 1.7B,
   +0.5-1.0 at 14B (not significant) and −0.7 at 32B (p=0.38). Raw calibration improves (ECE 0.137 → 0.115, NLL 0.95 → 0.67)
   but after a temperature it does not beat zero-shot + T. The remaining 10 points to Jev are not closed by head training at
   this scale; the natural reading is backbone capability or a post-training (RLCD) orders of magnitude larger.
4. **The benefit of shared computation and the isolation do not depend on the backbone.** At Q=10 the three shared engines are
   7-9x naive, and leakage is ≈0 with a negative control of 0.995 at all three sizes. Absolute packed speed goes 48.6 → 9.3 →
   2.5 q/s (S=2k, Q=100), slowing roughly in proportion to the parameter count.
5. **The mask-free D3 gains more the larger the backbone.** shared / packed at S=2038, Q=100: 0.87x at 1.7B, 1.07x at 14B,
   **1.50x** at 32B. The more layers and heads, the more the waste of dense masked attention bites.
6. **Selective accuracy (32B, calibrated)**: at p ≥ 0.9, 44% of MMLU can be processed automatically at accuracy 0.97; at
   p ≥ 0.7, 75% at 0.91 (slot + perm_avg: 50% at 0.97).
7. The fifth-option Δlog-odds is −0.49 at 1.7B, −1.24 at 14B and +0.08 at 32B (all on the 21 bridge items, CI ±0.4), which is
   inconsistent; a re-measurement at n=300 is needed.

The next external evaluation is JevBench (Benchmark Heaven; Jev's hard tier is 74.1%), to see what the 11-point MMLU gap
becomes on decision-style tasks (`tasks/active/jevbench-eval.md`).

## External evaluation on JevBench (Benchmark Heaven v1.2, 231 public decisions)

[JevBench](https://github.com/fstandhartinger/jevbench) (MIT) is a benchmark built for Jev-style decision models: typed
decisions of three kinds (choice / noul / score) against a state plus a rubric. The public part is easy 48 / standard 72 /
hard 111 = 231 decisions (the judge tier of 146 and 303 held-out items are private and are measured by
[Benchmark Heaven](https://benchmarkheaven.com/jev-models) on request). The hard tier has 2-6k-token policies, multi-hop
reasoning, date and number reasoning, adversarial distractors and items whose correct answer is "no clear answer"; it is
closer to what a Decision API is for than MMLU.

jqv implements the TypeSafe-compatible `POST /v1/systemone` (`jqv/systemone.py`) and was run through the harness's `typesafe`
adapter unchanged (`scripts/jevbench_run.py`, one request at a time, local MPS, harness commit 7ce310c). JSON states (35 of the
public items) are given to the model as pretty-printed JSON text. The temperature fitted on MMLU val was reused as is, and
nine configurations were measured by the harness **with the server returning calibrated probabilities**
(`jevbench_run.py --temperature-file`). The same nine configurations were also run returning raw probabilities
(`results/jevbench/<label>/`, without `_T`): accuracy is identical, and the calibrated ECE measured by the harness equals the
value obtained by applying T to the raw probabilities afterwards (`results/jevbench_summary.md` has both sets of rows).

| run (server returns calibrated probabilities) | easy 48 | standard 72 | hard 111 | Intelligence (3 public tiers) | hard ECE | Calibration (ECE only) | raw p50 |
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
| Jev 1.13 (Benchmark Heaven v1.2.5, 534 decisions incl. held-out) | - | - | **0.741** | 90.4 | - | 82.7 (ECE + TVD) | 0.65 s |
| SemIf Qwen3.5-4B (same) | - | - | 0.595 | 85.9 | - | 72.6 | 0.20 s |
| openjev-sglang Qwen3.6-35B-A3B (same) | - | - | 0.714 | 88.9 | - | - | 0.68 s |
| GPT-5.6 Luna low (same) | - | - | 0.945 | 96.8 | - | 89.8 | 0.97 s |

Intelligence is the weighted accuracy over the three public tiers (easy 14 / standard 28 / hard 30, renormalised for the
missing judge tier), a different condition from the leaderboard value (judge + held-out included). Calibration is the top-label
ECE on the hard tier only (the 20 distribution-gold items are private). Latency is the raw, unadjusted value (Benchmark Heaven
adds ×2 + 0.15 s for self-hosted systems).

The 32B zero-shot hard tier by family (correct / items): adversarial 6/6, trap 8/8, routing 5/5, judge 14/17, versus
long_policy 9/19 (11/19 with perm_avg), multi_hop 10/18, temporal_numeric 5/15, probability 4/10, tradeoff 3/6.

What this shows:

1. **easy is 100% even at 1.7B; standard is 95.8% at 32B.** Simple classification / routing / extraction of an explicit answer
   saturates with a small backbone; the backbone difference shows on hard (1.7B 42.3 → 14B 55.0 → 32B 62.2%).
2. **The gap to Jev is the same width as on MMLU.** On the hard tier Jev has 74.1% against jqv-32B's 62.2% (perm_avg 64.0%), a
   10-12-point gap, about the same as the MMLU gap (10.9 points). The gap neither narrows nor widens on decision tasks. jqv is
   above SemIf Qwen3.5-4B (59.5%) and below openjev-sglang Qwen3.6-35B-A3B (71.4%); note that Jev's value is on the 220 items
   including held-out.
3. **The weak spots are applying long rules and numeric / temporal reasoning** (long_policy 47-58%, temporal_numeric 33%,
   probability 40-50%). The topics on which Jev leads the open models most on Benchmark Heaven are rules / policy / finance,
   consistent with Jev's post-training working in the direction of "read the rules and make a typed decision". Conversely,
   adversarial / trap / routing are perfect at 32B.
4. **With raw probabilities the Calibration axis collapses.** The raw hard-tier ECE is 0.27 even at 32B (0.54 at 1.7B), a
   Calibration score of 45 (0 at 1.7B). Merely serving the MMLU-val T brings the harness-measured ECE to 0.107 and Calibration
   to 78.6 (Jev 82.7, SemIf 72.6). A Decision API only gets onto the board once it returns calibrated probabilities. The
   temperature transfer from MMLU to JevBench hard is **partial**: the ECE improves a lot (0.274 → 0.107) but stays far from the
   in-MMLU 0.023 and does not meet the criterion of the transfer section (diagonal + 0.02). Three levels of distribution shift
   are visible: MMLU ↔ JMMLU (transfers strongly), MMLU → JevBench hard (partial), MMLU → bridge (harmful).
5. **Decision training (slot + LoRA) does not help here either.** 32B hard 62.2% (same as zero-shot), 14B 58.6% (+3.6).
   perm_avg gives +1.8 on the 32B hard tier.
6. **Speed**: the 32B raw p50 is 0.68 s (Jev 0.65 s, but Jev is a production API). With Benchmark Heaven's adjustment (×2 +
   0.15 s) that corresponds to 1.5 s. The 1.7B is at 0.04 s.

Caveats: public 231 items only (no judge tier and no distribution-gold items), English only, and the hard tier is a pilot
authored by Claude Opus 5 / GPT-5.6 Sol. The Jev row quotes Benchmark Heaven's measurement (all 534 items) and is not a
paired comparison on the same items.

### Measurement by Benchmark Heaven (JevBench v1.2.7, 2026-09-21)

The public endpoint below was submitted as a bench request
([jevbench#6](https://github.com/fstandhartinger/jevbench/issues/6)) and measured by Benchmark Heaven from Germany with the
`typesafe` adapter unchanged, one request at a time. The result is published in
[JevBench v1.2.7](https://github.com/fstandhartinger/jevbench/blob/v1.2.7/RESULTS-v1.2.md) and on
[benchmarkheaven.com/jev-models](https://benchmarkheaven.com/jev-models) as a **partial row (shown but not ranked)**. The
measurement conditions and cost basis were committed before the run in
[`docs/v1.2-additions-run3.md`](https://github.com/fstandhartinger/jevbench/blob/v1.2.7/docs/v1.2-additions-run3.md)
(copies, together with the published row's JSON, are in `results/jevbench/published_v1.2.7/`; the maintainer's full
comments are in `results/public/jevbench_issue6_comments.md`).

| item | Benchmark Heaven's measurement | our own harness run (public items only) |
|---|---:|---:|
| easy (72 items incl. held-out) | 72/72 = 1.000 | 1.000 (48 items) |
| standard (96 items incl. held-out) | 92/96 = 0.958 | 0.958 (72 items) |
| judge (146 items, private) | 135/146 = 0.925 | not measured |
| hard public (111 items) | 69/111 = 0.622 | 0.622 |
| hard held-out (109 items) | not sent, counted as unanswered | private |
| hard tier (aggregated over 220 items) | 0.314 | - |
| hard ECE / Brier | 0.107 / 0.515 | 0.107 / - |
| Calibration axis | 74.9 (ECE plus probability fidelity 71.1) | 78.6 (ECE only) |
| p50 / p95 (standard + judge, from Germany, raw) | 0.92 s / 4.71 s | 0.65 s (local) |
| Speed axis | 67.6 (p50 1.85 s after the ×2 self-hosted adjustment) | - |
| Cost axis | 52.8 ($0.0374 per 1,000 decisions, estimated at OpenRouter's `qwen/qwen3-32b` input tariff) | - |
| JevBench Score | **67.2** (Intelligence 76.1, no rank) | - |

What this shows:

1. **The self-measured public numbers were reproduced exactly** (easy 1.000, standard 0.958, hard 0.622). The configuration
   serving calibrated probabilities, the prompt hash and the temperature were confirmed on their side from the `/health` response.
2. **The judge tier (146 private items) is at 92.5%.** On the same tier Jev 1.13 scores 94.5% and classifier.dev (batched Jev)
   97.3%; like standard, an area where the backbone difference is small.
3. **On hard, the 109 held-out items were not sent, so the row covers 425 of 534 decisions and carries no rank.** The policy of
   not sending held-out items to a submitter-operated endpoint was adopted during the measurement (easy / standard / judge had
   already been sent before that decision). A rankable row needs public weights or serving code, or a production endpoint
   (the maintainer's comment). The serving code has since been published at
   [Octalab-Inc/jqv](https://github.com/Octalab-Inc/jqv) and a re-run was requested in
   [jevbench#9](https://github.com/fstandhartinger/jevbench/issues/9).
4. **Calibration 74.9 is lower than the ECE-only self-measurement of 78.6.** Benchmark Heaven also includes the probability
   fidelity on the distribution-gold items (71.1); most of the difference to Jev's 82.7 shows up there.
5. **Speed and Cost depend heavily on the evaluation conditions.** The raw p50 of 0.92 s is the local 0.65 s plus the
   Japan-Germany round trip, and corresponds to 1.85 s after the ×2 self-hosted adjustment. Cost is estimated at the base
   model's public tariff even for a free endpoint. Competing on the Speed axis would require a CUDA server near Europe.

### Ranked re-run by Benchmark Heaven (JevBench v1.2.8, 2026-09-21)

After the serving code was published ([Octalab-Inc/jqv](https://github.com/Octalab-Inc/jqv), commit 0189b67, Apache-2.0),
a second bench request ([jevbench#9](https://github.com/fstandhartinger/jevbench/issues/9)) asked for a run on the
maintainers' own hardware. They ran the same configuration (`/health`: prompt_hash 4f85a0b34776, temperature
3.0225814579771493, Qwen3-32B BF16, packed engine) on a RunPod H100 NVL 96 GB (torch 2.11 + CUDA 12.8) in Canada, called
from their server in Germany one request at a time, over all 534 decisions including the 109 held-out hard items. The
partial row was replaced by a complete, ranked one ([v1.2.8](https://github.com/fstandhartinger/jevbench/tree/v1.2.8);
copies of the published row in `results/jevbench/published_v1.2.8/`, the comments in `results/public/jevbench_issue9_comments.md`).

| item | v1.2.8 (maintainers' H100, all 534 decisions) | v1.2.7 (tunnel to our Mac, 425 of 534) |
|---|---:|---:|
| rank / JevBench Score | **#8 of 36 / 70.1** | not ranked / 67.2 |
| Intelligence / Calibration / Speed / Cost | 86.1 / 79.0 / 74.6 / 47.5 | 76.1 / 74.9 / 67.6 / 52.8 |
| easy / standard / judge | 1.000 / 0.958 / 0.925 | 1.000 / 0.958 / 0.925 |
| hard (220 items) | 0.645 (142/220) | 0.314 (69 correct, 109 unanswered) |
| hard public 111 / held-out 109 | 0.622 (69/111) / 0.670 (73/109) | 0.622 / not sent |
| hard ECE / Brier / probability fidelity | 0.088 / 0.472 / 75.5 | 0.107 / 0.515 / 71.1 |
| p50 / p95 (standard + judge, from Germany, raw) | 0.75 s / 0.97 s (adjusted 1.64 s) | 0.92 s / 4.71 s (1.85 s) |
| cost per 1,000 decisions | $0.0564 est. | $0.0374 est. |

Hard tier by family on all 220 items (correct / items): adversarial 12/12, trap 16/16, routing 9/10, judge 26/33,
tradeoff 8/12, multi_hop 24/35, ambiguous 9/14, probability 12/20, long_policy 19/38, temporal_numeric 7/30.

What this shows:

1. **The public tiers reproduced a third time** (easy / standard / judge unchanged), and the held-out hard items came out
   slightly above the public ones (67.0% vs 62.2%), so using the public tiers once as a development gate did not inflate
   the number.
2. **The hard-tier gap to Jev 1.13 is 9.6 points** (74.1 vs 64.5), the same width as on MMLU (10.9). The weak families are the
   same on the held-out half: temporal_numeric 7/30 (2/15 on the held-out items), long_policy 19/38 and probability 12/20,
   against adversarial / trap / routing at or near 100%. These are the families the synthetic training data targets.
3. **Calibration 79.0** (Jev 82.7): the hard-tier ECE is 0.088 and the probability fidelity 75.5, both better than in the
   tunnel run (0.107 / 71.1) because the held-out items are now included, not because the model changed.
4. **Speed 74.6** from a raw p50 of 0.75 s across the Atlantic on an H100 (×2 + 0.15 s adjustment = 1.64 s); our local MPS
   p50 was 0.65 s. The cost estimate rose to $0.0564 because the held-out hard items are long (359 input tokens per
   decision on average over the whole set).
5. The run also confirms the CUDA path of the `packed` engine (torch 2.11 + CUDA 12.8), which we had not exercised ourselves.

## Public endpoint (Cloudflare Quick Tunnel)

The 303 held-out items and the judge tier are measured by Benchmark Heaven on a bench request. Because jqv's `/v1/systemone` is
TypeSafe-compatible, exposing it over HTTPS makes it measurable with the harness's `typesafe` adapter (`--key-env ''`) as it is.
The temporary-exposure procedure:

```bash
brew install cloudflared
caffeinate -dimsu &                                                    # keep the machine awake during the measurement
JQV_MODEL=Qwen/Qwen3-32B JQV_ENGINE=packed \
JQV_TEMPERATURE_FILE=results/mmlu_packed_qwen3-32b_temperature.json \
uv run uvicorn jqv.server:app --host 127.0.0.1 --port 8000 &          # the configuration that returns calibrated probabilities
cloudflared tunnel --url http://localhost:8000                          # issues https://<random>.trycloudflare.com
curl -s https://<random>.trycloudflare.com/health                       # check that calibration.temperature is reported
```

- A Quick Tunnel needs no account or DNS setup; the URL is random and disappears when `cloudflared` stops. While it is up,
  anyone who knows the URL can call it, so stop it right after the measurement.
- Benchmark Heaven calls from Germany one request at a time, so the external p50 includes network latency (the ranking adds
  ×2 + 0.15 s for self-hosted systems). Measured from the same Mac through the public URL, requests of 0.23-0.74 s locally
  gained +0.04-0.12 s through the tunnel (via a Cloudflare edge in Japan); from Germany the round trip adds more. Competing on
  the Speed axis would mean Linux/CUDA near Europe.
- What was submitted: 32B zero-shot + T=3.0 (hard 0.622, Calibration 78.6, p50 0.65 s). perm_avg gains +1.8 pt at twice the
  p50, a net loss on the score.
- The bench request was filed on 2026-09-21
  ([fstandhartinger/jevbench#6](https://github.com/fstandhartinger/jevbench/issues/6), 32B zero-shot + T=3.02, repository
  still private at the time; the text is in `results/public/bench_request_issue.md`), measured the same morning and published
  in JevBench v1.2.7 (section above). The tunnel was taken down at 08:51 after the comment "measured, you can take the tunnel
  down". cloudflared's counters showed 541 requests in total while public (511 with status 200, 2 with 400 from our own probes,
  27 with 404 from path scanning).
- 2026-09-21: the serving code was published as [Octalab-Inc/jqv](https://github.com/Octalab-Inc/jqv) and a rankable re-run
  including the held-out items was requested in [jevbench#9](https://github.com/fstandhartinger/jevbench/issues/9); the
  maintainers ran it the same day on their own H100 (instructions in `docs/jevbench-serving.md`), see "Ranked re-run" above.
- Monitoring while public: `scripts/watch_public.sh` (every 5 minutes it logs issue comments, health through the tunnel and on
  localhost, and the request-count deltas from cloudflared's `/metrics` to `results/public/watch_public.log`, and restarts any
  process that has disappeared; it never writes to GitHub). Because the uvicorn access log pointed at the git-tracked
  `results/public/server.log`, a branch switch replaced the file and the log of the measurement was lost;
  `results/public/*.log` is now ignored.

## Synthetic data for the weak hard families (`jqv/synth/`, `data/synth/`)

The JevBench families where the 32B is weakest are long_policy (9/19), temporal_numeric (5/15) and probability (4/10).
Targeted training needs items whose answers are certain, so every synthetic item's answer comes from a solver (a rule
engine, calendar arithmetic with `datetime` / `zoneinfo`, exact probabilities with `fractions`); a language model is used
only to vary the wording of narrative paragraphs. The items have the JevBench shape (state, typed question `choice` /
`noul` / `score` with criteria, labels, expected) and live in `data/synth/<family>/{train,dev,test}.jsonl` (2,000 / 300 /
500 per family; dev and test are committed, train is regenerated with seed 0 and the saved paraphrase patch).

- **long_policy**: five rule-engine domains (homeowners water damage, commercial equipment breakdown, trip cancellation,
  an HR relocation-reimbursement policy, SLA service credits). Each document (1.3-3.0k tokens, median 2.25k) has definitions,
  numbered exclusions with exceptions, dated endorsements or amendments that change a sublimit or threshold, generic
  clauses, a correspondence excerpt and a claim file, plus a trainee note that argues for the surface answer; the decision
  labels are computed by executing the same rules.
- **temporal_numeric**: six scenarios (month-end and leap-year rules with time zones, business-day deadlines with holidays
  and an after-hours rule, continuous-service months with leave resets, pro-rated invoices, DST cut-offs, unit conversions
  against a cap). A note in the state presents a wrong computation.
- **probability**: six scenarios (hypergeometric acceptance sampling with superseded plan versions, screening posteriors with
  age-group prevalence, k-of-n redundancy, expected-value choice, supplier mix with Bayes, draw outcomes). `noul` items about
  a random event carry the true P(yes) as `target_distribution`, and draw-outcome `choice` items carry the full outcome
  distribution, for the later proper-scoring experiment.
- **multi_hop as an attribute, not a family**: the solver's derivation trace gives every item `dependency_hops` (the number of
  derived intermediate facts) and `reasoning_depth` (the longest derivation chain); hops range from 2 to 7.
- **Distractors**: generation prefers items in which the note's shortcut gives a different answer from the truth (60% of the
  temporal_numeric and 54% of the probability dev items), the failure mode the JevBench hard items are built around.
- **Contamination**: 0 shared word 8-grams with the 231 JevBench public items and no reused IDs or invented names
  (`scripts/synth_contamination.py`); getting there required rewording several standard-form phrases and the label vocabulary
  that the first drafts had copied from the JevBench examples.

Difficulty check (`scripts/synth_difficulty.py`, `results/synth_difficulty.md`; zero-shot, packed, dev 300 per family):

| family | 14B dev | 32B dev | JevBench 32B (target ±10 pt) |
|---|---:|---:|---:|
| long_policy | 0.333 | 0.383 | 0.47 (9/19) |
| temporal_numeric | 0.270 | 0.323 | 0.33 (5/15) |
| probability | 0.447 | 0.453 | 0.40 (4/10) |

The first generation was too easy for temporal_numeric (14B 0.507) and probability (14B 0.657); two hardening rounds
(distractors that disagree with the truth, fewer two-option questions, per-scenario redraws, wider parameter spaces) brought
all three families inside the target band at 32B. The models follow the wrong note in most items where it disagrees with
the truth (surface-answer rate 0.6-1.0 in temporal_numeric). Accuracy is not monotone in hops: in long_policy the 6-hop
items (the "pay within the sublimit" decisions) are the easiest, so hops measure the length of the derivation rather than
difficulty by itself. Surface variation: Qwen3-14B rewrote one fact paragraph per train item (claim files, reports, records only; rules, clauses and the distractor notes are never rewritten), and a rewrite was kept only if every number, date, identifier and name survived, the length stayed within 0.6-1.6x, and every comparison and negation word kept its count. Within the 2-hour box this changed 612 long_policy (31%), 568 temporal_numeric (28%) and 125 probability (6%) train items; the model returned the paragraph unchanged in about half of the accepted cases, and those were dropped. The accepted rewrites are stored as `train.paraphrase.jsonl` patches and re-applied when train is regenerated; dev and test are untouched.

Tests: `tests/test_synth.py` (25 tests) checks the solvers on hand-computed cases through a `facts` override of every
scenario, the trace bookkeeping, item invariants, split deduplication and the loader round trip.

## Targeted LoRA on the weak families (14B gate, then 32B on GB10)

The slot head + LoRA trained on MMLU alone did not move accuracy at 14B or 32B (section C, backbone scaling). The same head and LoRA were
retrained on a mixture of the synthetic hard-family data (previous section; 2,000 training items per family) and MMLU, first at 14B as a gate,
then at 32B with the identical recipe. Setup: slot head + LoRA r=16, CE only (λ=0), 600 steps × batch 8 (micro-batch 2 × 4 accumulation, gradient
checkpointing), max_len 4096, `--train-mix synth:long_policy=0.25,synth:temporal_numeric=0.25,synth:probability=0.2,mmlu=0.3` (one source per batch,
so 300-token MMLU items and 3k-token policies never share a batch), validation on 96 synthetic dev items per family + 64 MMLU val items every 100 steps.
Evaluation: paired comparison (`scripts/compare_runs.py`, exact McNemar + bootstrap CI) on the synthetic test sets (500 per family) and MMLU / JMMLU test
800 against the zero-shot vocabulary readout, plus JevBench public hard (111 decisions, served temperature). The 32B run was trained on a GB10
(DGX Spark class, 121 GB unified memory, CUDA 13) instead of the Mac: 501 min for 600 steps (50 s/step, 1.75× the MPS speed), see `docs/gb10.md`.

| Evaluation | 14B zero-shot | 14B targeted LoRA | Δ [95% CI] | 32B zero-shot | 32B targeted LoRA | Δ [95% CI] | McNemar p (32B) |
|---|---:|---:|---:|---:|---:|---:|---:|
| synth long_policy test (500) | 0.356 | **0.566** | +0.210 [+0.158, +0.266] | 0.380 | **0.596** | +0.216 [+0.164, +0.274] | <0.001 |
| synth temporal_numeric test (500) | 0.272 | **0.694** | +0.422 [+0.364, +0.482] | 0.302 | **0.668** | +0.366 [+0.306, +0.428] | <0.001 |
| synth probability test (500) | 0.482 | **0.752** | +0.270 [+0.214, +0.330] | 0.468 | **0.766** | +0.298 [+0.242, +0.352] | <0.001 |
| MMLU test (800) | 0.750 | 0.760 | +0.010 [−0.007, +0.028] | 0.806 | 0.821 | +0.015 [+0.001, +0.030] | 0.058 |
| JMMLU test (800) | - | - | - | 0.767 | 0.779 | +0.011 [−0.004, +0.028] | 0.20 |
| JevBench public hard (111, served T) | 0.550 (61) | 0.604 (67) | +0.054 (16 won / 10 lost, p=0.33) | 0.613 (68) | 0.649 (72) | +0.036 (13 won / 9 lost) | 0.52 |

The 32B zero-shot baselines were re-measured on the GB10 (`results/*_gb10*`); the Mac numbers (MMLU 0.809, JMMLU 0.771, JevBench 69/111) differ only by
bf16 hardware noise: 110 of the 111 JevBench predictions are identical, and the one flip (`hard-opus-a-temporal_numeric-07`) is a near-tie between two
dates (0.335 vs 0.321 on the Mac, 0.311 vs 0.338 on the GB10). Seven of the 111 items have a top-2 margin below 0.05, so ±1–2 items across hardware or
dtype is expected.

Calibration (raw probabilities, T=1, 32B): synthetic test ECE drops from 0.53 / 0.61 / 0.40 (zero-shot, mean confidence ≈ 0.9) to 0.075 / 0.062 / 0.053
(long_policy / temporal_numeric / probability); MMLU raw ECE 0.139 → 0.093 and the fitted temperature 3.0 → 1.75 (the trained head internalises the
temperature, as in section E); NLL after temperature improves on MMLU (0.535 → 0.506) and JMMLU (0.609 → 0.587). On JevBench hard the served ECE goes
0.127 → 0.096, Brier 0.516 → 0.415, ordinal MAE 0.77 → 0.56, macro accuracy 0.673 → 0.713. Selective accuracy (32B, LoRA): probability answers 50 % of the
items at 0.92 accuracy when p ≥ 0.9; temporal_numeric 59 % at 0.76 when p ≥ 0.7; long_policy 42 % at 0.78 when p ≥ 0.7.

JevBench hard by family (correct / items, zero-shot → LoRA): at 32B long_policy 9 → 10 / 19, multi_hop 10 → 11 / 18, temporal_numeric 4 → **2** / 15,
probability 4 → **8** / 10, ambiguous 5 → 6 / 7, trap 8 → 7 / 8, the rest unchanged; at 14B long_policy 5 → 8, temporal_numeric 5 → **3**, probability 5 → 6,
tradeoff 1 → 3, ambiguous 4 → 5, trap 7 → 8. The three temporal items lost at 32B are an EUR amount computation, a 30-month cap expiry and a deadline
yes/no: all require the numeric or date arithmetic that decides the choice. On the synthetic test the 6-hop long_policy items (paying within a sub-limit,
n=30) get worse at both sizes (0.60 → 0.33 at 32B, 0.57 → 0.27 at 14B), while the rate of following the misleading note falls from 0.80 to 0.20 (3-hop items).

What this shows:

1. **Targeted LoRA is selective transfer, not a general lift of "hard" ability.** In-distribution gains are +22 to +42 points (p<0.001), but on the same
   111 JevBench items the paired result is 13 won / 9 lost (exact McNemar p=0.52) at 32B and 16 / 10 (p=0.33) at 14B. The stronger claim today is that the
   decision distribution improves (ECE, Brier and ordinal MAE all move the same way), not that accuracy does.
2. **Transfer is heterogeneous by family, with the same pattern at 14B and 32B.** Probability transfers positively (4 → 8 / 10, n=10, so an independent
   test set is needed). Long-policy training succeeds in-distribution (+21.6 points) but transfers weakly (+1 item). Temporal-numeric training shows a
   reproducible negative transfer (5 → 3 and 4 → 2). Multi-hop, judge, adversarial and routing items are unaffected, so nothing was broken globally.
3. **The temporal generator has to be redesigned, not enlarged.** The lost items need contract period + cap period, amount + currency + threshold,
   effective date + expiry conditions, before/after deadline booleans and AND/OR over several numeric conditions, i.e. the computation that decides the
   choice; the current generator centres on month-end rules, business days, DST and unit conversion (`tasks/active/temporal-generator-v2.md`).
4. **MMLU / JMMLU do not regress** with a 30 % replay; the mixed run even beats the MMLU-only 32B head (0.821 vs 0.801).

Conclusion: targeted LoRA improves in-distribution hard-family performance and probability quality, but transfer to JevBench is heterogeneous.
Probability reasoning transfers positively, long-policy transfer is limited, and temporal-numeric training shows reproducible negative transfer across
14B and 32B. Generic targeted fine-tuning is therefore insufficient; the next iteration redesigns the temporal-numeric generator around the computation
patterns observed in the held-out-style errors rather than simply increasing data volume.

## Related projects

Several public implementations arrived independently at the same hypotheses in 2026 (read the option-token logits directly
without generating, prefill a shared state once, trained heads, Brier training, shared-prefix attention). URLs were checked on
2026-09-20. The numbers are the self-reported values in each repository's README; benchmarks and conditions differ from jqv.

| project | closeness to jqv | what it does | difference from jqv |
|---|---|---|---|
| [featherless-ai/simple-jev](https://github.com/featherless-ai/simple-jev) | ★★★★★ | Reads the answer label from the next-token logits of the prefill; reuses the KV cache of the common prefix for a batch of question suffixes; RFDT (trains the answer-token logits directly, LoRA) | The same idea as jqv's naive → kvcache (B → D1). Its README states that the softmax values are not calibrated probabilities of correctness. jqv goes on to packed / shared (D2 / D3) and to measuring calibration |
| [TianyuCodings/NanoJev](https://github.com/TianyuCodings/NanoJev) | ★★★★★ | Qwen3-0.6B + a dedicated decision head; CE / Brier training, observed-event probability, paired proper reward, RLCD-style experiments; compares against the true probability q in simulators (maze / Snake) | The same theme as jqv's C / E. NanoJev uses controlled environments where the true q is available; jqv uses general semantic decisions on MMLU / JMMLU / bridge. Its evaluation policy of looking at NLL / Brier, risk-coverage and OOD separately, not only ECE, corresponds to jqv's `compare_runs.py` (selective accuracy) and the temperature-transfer table |
| [TheoLeeCJ/SemIf](https://github.com/TheoLeeCJ/SemIf) (formerly OpenJev) | ★★★★☆ | Qwen3.5-4B, runtime-defined criteria, direct option logits, shared-state prefill; compared on its own benchmark and the public TypeSafe subset (self-reported) | States that its scores are not operationally calibrated the way Jev's are; an issue discusses whether the best temperature differs per task. jqv has measured exactly that (transfers across languages, not across task types) |
| [ekzhang/openjev-sglang](https://github.com/ekzhang/openjev-sglang) | ★★★★☆ | Qwen3.6-35B-A3B + SGLang radix cache behind a Jev-compatible API (choice / noul / rubric, prefill only); verified with TypeSafe's official SDK | Ahead of jqv on the "production serving" phase. States that its calibration is a probability conditioned on the supplied options, not a calibrated estimate of correctness |
| [r-ms/mini-jev](https://github.com/r-ms/mini-jev) | ★★★★☆ | A pre-registered experiment reading option-letter logits from a frozen Qwen3-4B | The same readout as jqv's B; no training, shared computation or calibration |
| [zwliJay/jev-forge](https://github.com/zwliJay/jev-forge) | ★★★☆☆ | A training and inference stack that scores dynamic candidate branches from a shared prefix (high cardinality, calibration, batched inference) | Close as a training stack; jqv leans towards engine ablations and calibration measurement |
| [bnsd55/jevmlx](https://github.com/bnsd55/jevmlx) | ★★★★☆ | A practical Decision API for Apple Silicon / MLX: prefills the context + schema catalogue once and scores each field's options as trie rows against a broadcast KV cache in one batched forward; schema-constrained JSON, CalibrationBundle, timing ledger | Its execution structure is closest to jqv's D1 (shared KV). The structured-prediction side (schema, trie, multi-field) is more developed than jqv's; it does not do the D2 / D3 architecture ablation |
| [Hydragen](https://arxiv.org/abs/2402.05099) (Juravsky et al., 2024) | systems | Computes the attention to a shared prefix for all suffix queries together, sharing the KV reads; up to 32x on CodeLlama-13B | jqv's D3 (`jqv/engine/shared.py`) is a PyTorch implementation of this decomposition. On MPS it needs a second SDPA for the probe, so there are conditions where it loses to dense packed |
| [DeFT](https://arxiv.org/abs/2404.00242) (Yao et al., ICLR 2025) | systems | Flash tree-attention for tree-structured inference; cuts the IO of the shared KV by 73-99% | The direction beyond a CUDA implementation of jqv's packed / D3 (FlexAttention BlockMask) |

Other implementations of the same kind: [cobanov/awesome-jev](https://github.com/cobanov/awesome-jev) (a list of Jev-related
projects), [rongxinzy/LightJev](https://github.com/rongxinzy/LightJev) and [shamazharikh/qwen-rlcd](https://github.com/shamazharikh/qwen-rlcd).

**Applications / downstream** (users of Jev rather than reimplementations): [classifier.dev](https://classifier.dev/)
([mrmps/classifier-dev](https://github.com/mrmps/classifier-dev)) offers bulk classification over HTTP / CLI / MCP with
TypeSafe's Jev as the backbone and falls back to an LLM when Jev is unavailable (up to 20 decisions). jqv's `/v1/systemone`
can be called by such clients with the same wire format.

What is specific to jqv:

1. The engines are decomposed into A / B / B' / D1 / D2 / D3, with tests guaranteeing that the choice logits agree in fp32,
   and the speeds compared on that basis (the others stop at a shared KV; there is no systematic comparison of a packed block
   mask against a causal negative control).
2. Hume's secret-code isolation is reproduced with a negative control (packed 0.000, packed_causal 0.996).
3. On general semantic benchmarks (MMLU / JMMLU / bridge), raw → temperature → slot + LoRA → CE + λ·Brier are compared on the
   same test set and reported with paired comparisons and selective accuracy.
4. The domain transfer of the temperature (transfers across languages, not across task types) is measured.

NanoJev's framing, that RL is the tool for the case where only sampling / interaction is available and that with logits and
ground truth it is more natural to backpropagate the Brier score directly, agrees with jqv's E (try CE + λ·Brier first). In
jqv's results λ ≤ 1 is indistinguishable from CE and cross-distribution calibration is solved by neither the temperature nor
the Brier term, so an RLCD-type approach (outcome-based proper scoring) is the next candidate.

## Design notes

- **Separate tokenisation of prefix and suffix**: the state side and the question side are tokenised separately, and every
  engine uses the same token id sequence. A test checks that no BPE merge happens across the boundary (`tests/test_prompt.py`).
- **`kvcache` memory**: HF's `DynamicCache` materialises the prefix K/V per batch row, so for long states the batch is limited
  automatically by `cache_budget_bytes`. `packed` holds one copy of the prefix K/V and does not have this problem. vLLM's
  prefix caching (paged KV) shares the former; on Linux/CUDA it could replace `kvcache`.
- **Long packed sequences**: beyond `PackedEngine(max_tokens=...)` the prefix goes into the cache and only the question side is
  packed in chunks (same mask, query rows = questions only).
- **The cost of an explicit mask**: passing a 4D mask disables SDPA's causal fast path and is about 2x slower on 8k sequences,
  so no mask is passed when there is a single branch (packed) or no padding is needed (naive).
- **packed's block sparsity is not used to cut computation (a limit of the reference implementation)**: HF's SDPA / eager
  backends take the raw `(1, 1, L, L)` 4D mask and compute the attention densely before masking. What is shared is the
  prefix's linear layers (QKV/MLP) and its K/V; the attention matrix itself is built at L². At S=8038, Q=100 (L ≈ 13.5k) the
  ideal ratios are 60x for the linear layers and 36x for attention, and the measured 53x falls between them. At Jev's scale
  (a 23k-token state × 5,000 questions, L ≈ 173k) the mask alone would be 60 GB in bf16 and the attention FLOPs about 8x the
  ideal, so this implementation does not scale there. An implementation that uses the sparsity physically (a Hydragen-style
  decomposition that attends to the shared prefix for all branches together; FlexAttention's BlockMask on CUDA) is handled
  separately as D3.
- **bench.py appends to a JSONL per condition and skips completed conditions on re-run.** Stopping midway loses only the one
  condition in progress.
- **Calibration**: the raw logits of the 1-token readout are very sharp (mean confidence ≈ 0.96, ECE ≈ 0.38 on MMLU), and
  temperature scaling alone brings the ECE to around 0.06. The first step towards "0.8 means right 8 times out of 10" is
  reachable without modifying the model.

## Open issues

- **A proper evaluation of RLCD / proper scoring** — CE + λ·Brier did not beat temperature scaling (section E). Keeping
  calibration across distributions points to post-training with outcome-based proper scoring as the next candidate.
- **CUDA / FlexAttention / Hydragen-style serving** — on MPS, D3 needs two SDPA passes for the probe and stays at 1.5x packed
  at 32B; on CUDA it can be one block-sparse kernel. Serve the same `/decision` and `/v1/systemone` on top of vLLM / SGLang
  prefix caching.
- **Domain-shift calibration** — MMLU ↔ JMMLU transfers, JevBench hard transfers partially, bridge is harmed. Per-task-type
  temperatures, or calibration training that does not rely on a temperature.
- **The hard-tier gap to Jev** — ranked in v1.2.8 (#8 of 36, hard 64.5% vs Jev 74.1%); the gap sits in temporal_numeric,
  long_policy and probability, which the synthetic targeted training addresses next. Limited use of few-shot / perm_avg on the
  weak families is the training-free alternative.
- **Real applications** — bulk classification of the classifier.dev kind and grep-style routing on top of `/v1/systemone`,
  using selective accuracy (44% at accuracy 0.97 with p ≥ 0.9) as the operating metric.
