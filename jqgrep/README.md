# jqgrep — semantic code search with jqv (no index, no embeddings)

`jqgrep` answers a natural-language search intent with `path:start-end` line ranges, scored by jqv. It walks the live file
tree every time, keeps no index and no embeddings, and spends the model budget where jqv's shared-state inference pays off:
many short decisions against one long state.

```sh
uv run python -m jqgrep "where is the JWT signature verified" src/
uv run python -m jqgrep "retry logic around the HTTP client" . --model Qwen/Qwen3-14B
uv run python -m jqgrep "connection pool setup" . --server http://localhost:8000     # a running jqv server (POST /decision)
uv run python -m jqgrep "temperature fitting" . --json
```

```
query: where is the block attention mask for packed sequences built
files walked 4752 -> lexical 150 -> jqv routed 20 -> verified 20; ranges scored 61; model Qwen/Qwen3-14B engine packed; 143.0 s

0.91  jqv/engine/packed.py:38-96   [primary implementation; file 0.93]
      def build_block_mask(
          prefix_len: int,
          ...
```

## How it works

```
query, root
   │
   ▼  stage 0 — no model
walk (git ls-files / os.walk, .gitignore, no binaries, no huge files)
lexical score per file: query terms in content, path and top-level symbols, coverage of distinct terms,
a prior that favours source files over docs and discounts data / results / docs directories (--include-docs turns it off)
   │  150 candidates
   ▼  stage 1 — cheap semantic routing (jqv)
groups of 12 sketches (path, language, symbols, imports, head, lines with query words) share ONE state;
per file a short yes/no branch, plus one listwise "which of these files" question over the group;
score = p(yes) x (0.5 + 0.5 x listwise share / best); a final pass lets the best 2k files compete directly
   │  20 files
   ▼  stage 2 — full-file verification (jqv, one forward per file)
state = query + the whole file with line numbers (windows of ~7k tokens for long files);
questions = one yes/no per overlapping line range (40 lines, stride 30, <= 24 ranges),
            one listwise "which range" question,
            file relevant?  role: implementation / call site / test or example / config or docs
            production code?  ambiguous?
ranges above (best - gap) are merged, trimmed, ranked by range score x (0.6 + 0.4 x file score)
   ▼
0.91  src/auth/jwt.py:82-119   [primary implementation; file 0.93]
```

Stage 2 is where jqv's design matters: a 2-4k-token source file is prefilled once and 20-30 questions run as isolated
branches against it (`jqv.engine.packed` or `shared`), so asking for the role of the file or whether the match is
ambiguous costs almost nothing extra. Stage 1 uses the same shape with the sketches of a whole group as the shared state,
and the listwise questions make the model compare candidates instead of answering "yes" to each one in isolation.

## Ranking, not thresholds

jegrep advertises absolute yes/no probabilities so that a threshold "means something". jqv's calibration experiments show
that a temperature fitted on one distribution does not transfer to another task type, so `jqgrep` does not trust absolute
probabilities: it keeps the top-k and everything within `--gap` (default 0.25) of the best score at each stage. The
displayed scores are the (temperature-scaled when a fitted file exists) yes-probabilities combined with the listwise
share; `--threshold` adds an absolute cut for experiments, but it is off by default. A labelled code-search set would be
needed before an absolute threshold can be recommended.

## Options

| flag | default | meaning |
|---|---|---|
| `--model` | `$JQV_MODEL` or Qwen/Qwen3-1.7B | Qwen/Qwen3-14B is the recommended backbone; 32B for accuracy |
| `--engine` | packed | `shared` uses the mask-free D3 engine |
| `--server URL` | - | use a running jqv server instead of loading a model |
| `--lexical N` / `--top-files N` / `-k N` | 150 / 20 / 10 | candidates after stage 0 / files after stage 1 / hits returned |
| `--gap` | 0.25 | relative threshold at stages 1 and 2 |
| `--range-lines` / `--stride` / `--max-ranges` | 40 / 30 / 24 | line windows in stage 2 |
| `--max-state-tokens` | 7000 | files longer than this are split into overlapping windows |
| `--group-size` | 12 | sketches per shared state in stage 1 |
| `--include-docs` | off | do not discount docs / data files |
| `--skip-stage1` | off | send the lexical top files straight to stage 2 |
| `--json` | off | machine-readable output (hits with range, file and role probabilities) |

## Results on this repository

Three reference queries over this repository (4,751 files after the walk, 150 lexical candidates, default settings,
`--trace`). The expected file and its rank in the output:

| query | expected | Qwen3-14B | Qwen3-1.7B |
|---|---|---|---|
| where is the block attention mask for packed sequences built | `jqv/engine/packed.py` (`build_block_mask`) | **#1** `jqv/engine/packed.py:31-69` (0.91, file 0.98); the only file left after stage 1 (0.99; runner-up `README.md` 0.51) | #3 (0.55), behind two task notes |
| fitting the calibration temperature on the validation split | `scripts/fit_temperature.py` / `jqv/calibration.py` | **#1** `scripts/fit_temperature.py:31-68` (0.90, file 0.97); stage 1: 0.98, runner-ups `docs/jevbench-serving.md` 0.69, `scripts/transfer_temperature.py` 0.53 | #2 `jqv/calibration.py:20-80` (0.61), behind a task note |
| handling of the TypeSafe wire format (choice / noul / score questions) | `jqv/systemone.py` | **#1** `jqv/systemone.py:17-77` (0.77, file 0.97); stage 1: 0.97, runner-up `scripts/jevbench_run.py` 0.67 | not in the top 3 (task notes and the bench-request text rank higher) |

Wall time per query with the 14B on an Apple M5 Max: 145-173 s, of which loading the model is about 60 s (use `--server` to
load it once), stage 1 about 70 s (13 groups of 12 sketches plus a final pass over 40 files: 14 forwards, each a shared state
of a few thousand tokens with 12-40 short branches) and stage 2 about 10 s (one forward per verified file). The 1.7B takes 60-90 s.

What the runs show:

- With the 14B the stage-1 scores are sharp (0.97-0.99 for the right file, at most 0.69 for the rest), so the relative gap
  (0.25) leaves a single file for stage 2 on these single-implementation queries. For intents that are spread over several
  files (call sites, a feature touching many modules) raise `--gap` (0.5) or lower the bar with `--top-files`.
- The 1.7B answers "yes" to almost every sketch (0.7-0.8), so its ranking is carried by the listwise share and the lexical
  prior, and task notes in `tasks/` that quote the code outrank the code itself. That is why 14B is the recommended default.
- Stage 2 puts the range on the definition (`build_block_mask`, `fit`, the `choice` / `noul` / `score` translation), and the
  role question labels all three hits `primary_implementation`.

## Limits and next steps

- Scores are comparable within a query, not across queries or corpora (see "Ranking, not thresholds").
- Small backbones (1.7B) say "yes" to almost everything; the listwise questions help, but 14B is the practical minimum.
- Stage 0 is a heuristic lexical filter; a query with no lexical overlap falls back to the smallest files, which is weak.
  Sketch-only routing over all files (skipping the lexical filter) is possible but costs a forward per group of 12 files.
- Not done here: a labelled code-search benchmark comparing ripgrep/BM25, embeddings, jqv naive/packed/shared and Jev on
  Recall@k / MRR / latency, which would also give the calibration data needed for a meaningful `--threshold`.
