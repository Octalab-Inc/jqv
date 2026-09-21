"""jqgrep: semantic code search with jqv, no index, no embeddings.

    uv run python -m jqgrep "where is the JWT signature verified" src/
    uv run python -m jqgrep "retry logic" . --json --model Qwen/Qwen3-14B
    uv run python -m jqgrep "connection pool" . --server http://localhost:8000
"""

from __future__ import annotations

import argparse
import sys

from jqgrep.output import render_json, render_text
from jqgrep.search import SearchConfig, search


def main(argv=None):
    ap = argparse.ArgumentParser(prog="jqgrep", description="semantic code search with jqv (cascade: lexical -> jqv sketch routing -> shared-state range scoring)")
    ap.add_argument("query")
    ap.add_argument("path", nargs="?", default=".")
    ap.add_argument("--model", default=None, help="HF model id (default: $JQV_MODEL or Qwen/Qwen3-1.7B; Qwen/Qwen3-14B recommended)")
    ap.add_argument("--engine", default="packed", help="packed (default) | shared | kvcache | naive")
    ap.add_argument("--server", default=None, help="use a running jqv server (POST /decision) instead of loading a model")
    ap.add_argument("--device", default=None)
    ap.add_argument("--dtype", default=None)
    ap.add_argument("--no-temperature", action="store_true", help="rank on raw probabilities even if a fitted temperature file exists")
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--top-files", type=int, default=20, help="files kept after stage 1")
    ap.add_argument("--lexical", type=int, default=150, help="lexical candidates kept for stage 1")
    ap.add_argument("-k", type=int, default=10, help="hits returned")
    ap.add_argument("--gap", type=float, default=0.25, help="relative threshold: keep p >= best - gap (stages 1 and 2)")
    ap.add_argument("--threshold", type=float, default=None, help="optional absolute cut on range probability (not calibrated for code)")
    ap.add_argument("--range-lines", type=int, default=40)
    ap.add_argument("--stride", type=int, default=30)
    ap.add_argument("--max-ranges", type=int, default=24)
    ap.add_argument("--include-docs", action="store_true", help="do not discount docs / data files in the lexical stage")
    ap.add_argument("--group-size", type=int, default=12, help="sketches per shared state in stage 1")
    ap.add_argument("--max-state-tokens", type=int, default=7000)
    ap.add_argument("--skip-stage1", action="store_true", help="send the lexical top files straight to stage 2")
    ap.add_argument("--quiet", action="store_true")
    ap.add_argument("--trace", action="store_true", help="log the stage-1 ranking (top 30) to stderr")
    a = ap.parse_args(argv)

    if a.server:
        from jqgrep.jqv_client import HTTPClient

        client = HTTPClient(a.server)
    else:
        from jqgrep.jqv_client import LocalClient

        client = LocalClient(a.model, a.engine, a.device, a.dtype, temperature=None if a.no_temperature else "auto")
    cfg = SearchConfig(n0=a.lexical, n1=a.top_files, gap1=a.gap, k=a.k, gap2=a.gap, threshold=a.threshold, range_size=a.range_lines,
                       stride=a.stride, max_ranges=a.max_ranges, max_state_tokens=a.max_state_tokens, skip_stage1=a.skip_stage1,
                       code_only=not a.include_docs, group_size=a.group_size, trace=a.trace)
    log = (lambda m: None) if a.quiet else (lambda m: print(m, file=sys.stderr, flush=True))
    hits, stats = search(a.query, a.path, cfg, client, log)
    print(render_json(hits, a.query, stats) if a.json else render_text(hits, a.query, stats))


if __name__ == "__main__":
    main()
