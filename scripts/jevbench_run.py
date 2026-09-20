"""Run the JevBench harness (github.com/fstandhartinger/jevbench, MIT) against a local jqv server.

    uv run scripts/jevbench_run.py --model Qwen/Qwen3-1.7B --engine packed --label qwen3-1.7b_packed
    uv run scripts/jevbench_run.py --model Qwen/Qwen3-14B --engine packed --perm-avg --label qwen3-14b_permavg
    uv run scripts/jevbench_run.py --model Qwen/Qwen3-32B --engine slot --head-dir results/train/qwen3-32b_slot_best/best --label qwen3-32b_slot

Starts `uvicorn jqv.server:app` with the requested engine, waits for /health, runs the harness's `typesafe`
adapter (jqv serves the TypeSafe wire format at /v1/systemone) on each public tier, summarizes, stops the server.
Output: results/jevbench/<label>/<tier>/{results.jsonl, raw/, ledger.jsonl, manifest.json, summary.json}.
Latency is raw single-request wall time on this machine (Benchmark Heaven adds x2 + 0.15 s for self-hosted endpoints).
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

from _common import RESULTS

JEVBENCH = Path(os.environ.get("JEVBENCH_DIR", "/Users/h.imura/tmp/repo/jevbench"))
TIERS = {"easy": "easy.jsonl", "standard": "original.jsonl", "hard": "hard.jsonl"}


def wait_health(port: int, timeout_s: float) -> None:
    t0 = time.time()
    while time.time() - t0 < timeout_s:
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=5) as r:
                if json.loads(r.read()).get("ok"):
                    return
        except Exception:
            pass
        time.sleep(3)
    raise SystemExit("server did not become healthy")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--engine", default="packed")
    ap.add_argument("--perm-avg", action="store_true")
    ap.add_argument("--head-dir", default=None)
    ap.add_argument("--temperature-file", default=None,
                    help="serve temperature-scaled probabilities (JQV_TEMPERATURE_FILE; provenance-checked by the server)")
    ap.add_argument("--label", required=True, help="run directory name under results/jevbench/")
    ap.add_argument("--tiers", nargs="+", default=list(TIERS), choices=list(TIERS))
    ap.add_argument("--port", type=int, default=8010)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--force", action="store_true", help="delete an existing run directory")
    ap.add_argument("--load-timeout", type=float, default=1200)
    a = ap.parse_args()
    if not (JEVBENCH / "jevbench" / "cli.py").exists():
        raise SystemExit(f"harness not found at {JEVBENCH} (git clone https://github.com/fstandhartinger/jevbench)")
    run_dir = RESULTS / "jevbench" / a.label
    if run_dir.exists():
        clashing = [t for t in a.tiers if (run_dir / t).exists()]
        if clashing and not a.force:
            raise SystemExit(f"{run_dir} already has tiers {clashing}; pass --force to replace those tiers")
        for t in clashing:  # --force replaces only the selected tiers; other tiers' results are kept
            shutil.rmtree(run_dir / t)
    run_dir.mkdir(parents=True, exist_ok=True)
    harness_commit = subprocess.run(["git", "-C", str(JEVBENCH), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    env = {**os.environ, "JQV_MODEL": a.model, "JQV_ENGINE": a.engine, "JQV_PERM_AVG": "1" if a.perm_avg else "",
           **({"JQV_HEAD_DIR": a.head_dir} if a.head_dir else {}),
           **({"JQV_TEMPERATURE_FILE": a.temperature_file} if a.temperature_file else {})}
    env.pop("JQV_TEMPERATURE_FILE", None) if not a.temperature_file else None
    log = (run_dir / "server.log").open("w")
    server = subprocess.Popen([sys.executable, "-m", "uvicorn", "jqv.server:app", "--port", str(a.port)], env=env,
                              stdout=log, stderr=subprocess.STDOUT)
    try:
        print(f"[jevbench_run] starting server model={a.model} engine={a.engine} perm_avg={a.perm_avg} head={a.head_dir} "
              f"temperature={a.temperature_file}", flush=True)
        wait_health(a.port, a.load_timeout)
        print("[jevbench_run] server healthy", flush=True)
        cfg_path = run_dir / "config.json"
        cfg = json.loads(cfg_path.read_text()) if cfg_path.exists() else {}
        cfg.update({"model": a.model, "engine": a.engine, "perm_avg": a.perm_avg, "head_dir": a.head_dir,
                    "temperature_file": a.temperature_file, "harness_commit": harness_commit, "limit": a.limit,
                    "tiers": sorted(set(cfg.get("tiers", [])) | set(a.tiers))})
        cfg_path.write_text(json.dumps(cfg, indent=2))
        for tier in a.tiers:
            d = run_dir / tier
            d.mkdir()
            tasks = JEVBENCH / "datasets" / "public" / TIERS[tier]
            cmd = [sys.executable, "-m", "jevbench.cli", "run", "--tasks", str(tasks), "--adapter", "typesafe",
                   "--endpoint", f"http://127.0.0.1:{a.port}", "--key-env", "", "--model", a.label,
                   "--results", str(d / "results.jsonl"), "--raw-dir", str(d / "raw"), "--ledger", str(d / "ledger.jsonl"),
                   "--reserve-usd", "0", "--cost-basis", "local_mps_no_billing", "--manifest", str(d / "manifest.json"),
                   "--run-label", a.label]
            if a.limit:
                cmd += ["--limit", str(a.limit)]
            print(f"[jevbench_run] tier={tier} tasks={tasks.name}", flush=True)
            t0 = time.time()
            subprocess.run(cmd, cwd=JEVBENCH, check=False)
            summ = subprocess.run([sys.executable, "-m", "jevbench.cli", "summarize", "--tasks", str(tasks),
                                   "--results", str(d / "results.jsonl")], cwd=JEVBENCH, capture_output=True, text=True)
            (d / "summary.json").write_text(summ.stdout)
            try:
                s = json.loads(summ.stdout)
                print(f"[jevbench_run] {tier}: accuracy={s.get('accuracy')} ece={s.get('ece')} "
                      f"latency={s.get('latency')} ({time.time() - t0:.0f}s)", flush=True)
            except json.JSONDecodeError:
                print(f"[jevbench_run] {tier}: summarize failed: {summ.stderr[-500:]}", flush=True)
    finally:
        server.terminate()
        try:
            server.wait(timeout=30)
        except subprocess.TimeoutExpired:
            server.kill()
        log.close()
    print(f"[jevbench_run] done -> {run_dir}", flush=True)


if __name__ == "__main__":
    main()
