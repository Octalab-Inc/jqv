"""Engines that read decisions from a trained head (phase C) instead of the vocabulary.

    make_engine("pointer", rt, head_dir="results/train/pointer_lora/best")
    make_engine("slot", rt, head_dir=...)

`head_dir` holds `head/` (weights + config) and optionally `adapter/` (LoRA). Loading a LoRA adapter modifies
`rt.model` in place, so use a fresh Runtime per head when comparing several heads in one process.
"""

from __future__ import annotations

from pathlib import Path

import torch

from jqv.engine.base import DecisionEngine
from jqv.heads import load_head
from jqv.readout import _decision
from jqv.train.data import Example, collate, gather_features
from jqv.types import Decision, Question


class HeadEngine(DecisionEngine):
    name = "head"

    def __init__(self, rt, temperature=None, head_dir: str | Path | None = None, batch_size: int = 16, readout: str = "full",
                 perm_avg: bool = False):
        super().__init__(rt, temperature, readout, perm_avg)
        if head_dir is None:
            raise ValueError("head_dir is required (results/train/<run>/best or /last)")
        head_dir = Path(head_dir)
        self.head, self.head_cfg = load_head(head_dir / "head", rt.device)
        self.name = self.head.kind
        self.batch_size = batch_size
        if self.head_cfg.get("model") and self.head_cfg["model"] != rt.model_id:
            raise ValueError(f"head was trained for {self.head_cfg['model']}, runtime is {rt.model_id}")
        if self.head_cfg.get("prompt_hash") and self.head_cfg["prompt_hash"] != rt.prompt.hash:
            raise ValueError("head was trained with a different prompt layout")
        adapter = head_dir / "adapter"
        if adapter.exists() and not getattr(rt.model, "_jqv_adapter", None):
            from peft import PeftModel

            PeftModel.from_pretrained(rt.model, str(adapter))  # injects LoRA into rt.model in place
            rt.model._jqv_adapter = str(adapter)
            rt.model.eval()
        elif adapter.exists() and rt.model._jqv_adapter != str(adapter):
            raise RuntimeError(f"runtime already carries adapter {rt.model._jqv_adapter}; create a new Runtime")

    @torch.inference_mode()
    def _decide_plain(self, state: str, questions: list[Question]) -> list[Decision]:
        exs = []
        for q in questions:
            prefix = self.rt.prompt.prefix_ids(state, q.question, q.choices, q.labels)  # per question only for query-first layouts
            suffix, ends = self.rt.prompt.suffix_ids_with_spans(q.question, q.choices, q.labels)
            exs.append(Example(prefix + suffix, [len(prefix) + e for e in ends], 0))
        out: list[Decision] = []
        pad_id = self.rt.tokenizer.pad_token_id or 0
        for i in range(0, len(exs), self.batch_size):
            batch = collate(exs[i : i + self.batch_size], pad_id, self.rt.device)
            h = self.rt.backbone(input_ids=batch["ids"], attention_mask=batch["attn"], use_cache=False).last_hidden_state
            h_d, h_opts = gather_features(h, batch)
            z = self.head(h_d, h_opts, batch["opt_mask"]).float()
            for j, q in enumerate(questions[i : i + self.batch_size]):
                out.append(_decision(z[j, : len(q.choices)], self.temperature, None))
        return out
