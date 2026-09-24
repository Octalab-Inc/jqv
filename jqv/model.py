"""Model/tokenizer loading shared by all engines."""

from __future__ import annotations

import os
from dataclasses import dataclass

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from jqv.prompt import PromptBuilder, PromptStyle

DEFAULT_MODEL = os.environ.get("JQV_MODEL", "Qwen/Qwen3-1.7B")


def pick_device(device: str | None = None) -> torch.device:
    if device:
        return torch.device(device)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def pick_dtype(dtype: str | None, device: torch.device) -> torch.dtype:
    name = dtype or os.environ.get("JQV_DTYPE") or ("bfloat16" if device.type != "cpu" else "float32")
    return {"float32": torch.float32, "fp32": torch.float32, "bfloat16": torch.bfloat16,
            "bf16": torch.bfloat16, "float16": torch.float16, "fp16": torch.float16}[name]


@dataclass
class Runtime:
    model_id: str
    model: torch.nn.Module
    tokenizer: object
    device: torch.device
    dtype: torch.dtype
    prompt: PromptBuilder

    @property
    def lm_head(self) -> torch.nn.Module:
        return self.model.lm_head

    @property
    def backbone(self) -> torch.nn.Module:
        """The decoder stack (returns last_hidden_state after final norm)."""
        return self.model.model

    def sync(self) -> None:
        if self.device.type == "cuda":
            torch.cuda.synchronize()
        elif self.device.type == "mps":
            torch.mps.synchronize()


def default_style_for(model_id: str, layout: str | None = None) -> PromptStyle:
    """Chat template (thinking disabled) for instruct models, plain text for *-Base models."""
    return PromptStyle(chat=("base" not in model_id.lower()), layout=layout or "state_first")


def load_runtime(
    model_id: str | None = None,
    device: str | None = None,
    dtype: str | None = None,
    attn_implementation: str = "sdpa",
    style: PromptStyle | None = None,
    layout: str | None = None,
) -> Runtime:
    model_id = model_id or DEFAULT_MODEL
    dev = pick_device(device)
    dt = pick_dtype(dtype, dev)
    tok = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, dtype=dt, attn_implementation=attn_implementation)
    model.to(dev).eval()
    if style is None:
        style = default_style_for(model_id, layout)
    elif layout:
        from dataclasses import replace

        style = replace(style, layout=layout)
    return Runtime(model_id=model_id, model=model, tokenizer=tok, device=dev, dtype=dt, prompt=PromptBuilder(tok, style))
