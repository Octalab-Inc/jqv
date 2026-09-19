"""Train a decision head (+ LoRA) on MMLU auxiliary_train.

    uv run python -u scripts/train_head.py --head pointer --run-name pointer_lora --steps 600
    uv run python -u scripts/train_head.py --head slot --run-name slot_lora --steps 600
    uv run python -u scripts/train_head.py --head pointer --lora-rank 0 --run-name pointer_frozen
    uv run python -u scripts/train_head.py --run-name pointer_lora --resume      # continue from results/train/pointer_lora/last

Progress (loss, step/s, ETA) is printed every --log-every steps and appended to results/train/<run>/train_log.jsonl.
"""

from __future__ import annotations

import argparse

from _common import RESULTS, add_model_args, load_rt
from jqv.train.trainer import TrainConfig, Trainer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-name", required=True)
    ap.add_argument("--head", choices=["pointer", "slot"], default="pointer")
    ap.add_argument("--lora-rank", type=int, default=16, help="0 = frozen LLM, train the head only")
    ap.add_argument("--steps", type=int, default=600)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--head-lr", type=float, default=3e-4)
    ap.add_argument("--brier-weight", type=float, default=0.0)
    ap.add_argument("--max-len", type=int, default=512)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--log-every", type=int, default=10)
    ap.add_argument("--eval-every", type=int, default=100)
    ap.add_argument("--ckpt-every", type=int, default=100)
    ap.add_argument("--n-val", type=int, default=256)
    ap.add_argument("--pointer-rank", type=int, default=256)
    ap.add_argument("--no-shuffle-options", action="store_true")
    ap.add_argument("--grad-accum", type=int, default=1, help="micro-batches per step (effective batch unchanged)")
    ap.add_argument("--grad-checkpointing", action="store_true")
    ap.add_argument("--resume", action="store_true")
    add_model_args(ap)
    a = ap.parse_args()
    cfg = TrainConfig(run_name=a.run_name, head=a.head, lora_rank=a.lora_rank, steps=a.steps, batch_size=a.batch_size,
                      lr=a.lr, head_lr=a.head_lr, brier_weight=a.brier_weight, max_len=a.max_len, seed=a.seed,
                      log_every=a.log_every, eval_every=a.eval_every, ckpt_every=a.ckpt_every, n_val=a.n_val,
                      pointer_rank=a.pointer_rank, shuffle_options=not a.no_shuffle_options,
                      grad_accum=a.grad_accum, grad_checkpointing=a.grad_checkpointing)
    rt = load_rt(a)
    Trainer(rt, cfg, RESULTS / "train").train(resume=a.resume)


if __name__ == "__main__":
    main()
