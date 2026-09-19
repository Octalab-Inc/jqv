"""Prompt construction.

The prompt is split into a *prefix* (shared state) and a *suffix* (one question with its
choices, ending in the answer cue). Prefix and suffix are tokenized separately so the
same token ids can be used by every engine (naive concat, KV-cache branch, packed branch).
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"

DEFAULT_SYSTEM = (
    "You are a decision model. Read the document, then answer each question by choosing "
    "exactly one option. Reply with the option letter only."
)


@dataclass(frozen=True)
class PromptStyle:
    chat: bool = True  # Qwen chat template (thinking disabled) vs plain text (base model)
    system: str = DEFAULT_SYSTEM
    answer_cue: str = "Answer:"
    letter_prefix: str = " "  # readout token = letter_prefix + letter (" A", " B", ...)


class PromptBuilder:
    def __init__(self, tokenizer, style: PromptStyle | None = None):
        self.tok = tokenizer
        self.style = style or PromptStyle()
        self._choice_ids = self._resolve_choice_ids()

    # ----- text -----
    def prefix_text(self, state: str) -> str:
        doc = f"Document:\n{state}\n\n" if state.strip() else ""
        if self.style.chat:
            return (
                f"<|im_start|>system\n{self.style.system}<|im_end|>\n"
                f"<|im_start|>user\n{doc}"
            )
        return doc

    def suffix_text(self, question: str, choices: list[str]) -> str:
        if not 2 <= len(choices) <= len(LETTERS):
            raise ValueError(f"choices must have 2..{len(LETTERS)} entries, got {len(choices)}")
        opts = "\n".join(f"{LETTERS[i]}. {c}" for i, c in enumerate(choices))
        body = f"Question:\n{question}\n\nOptions:\n{opts}\n\nAnswer with the letter only."
        if self.style.chat:
            return (
                f"{body}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
                f"{self.style.answer_cue}"
            )
        return f"{body}\n{self.style.answer_cue}"

    # ----- ids -----
    def prefix_ids(self, state: str) -> list[int]:
        return self.tok.encode(self.prefix_text(state), add_special_tokens=False)

    def suffix_ids(self, question: str, choices: list[str]) -> list[int]:
        return self.tok.encode(self.suffix_text(question, choices), add_special_tokens=False)

    def choice_token_ids(self, n: int) -> list[int]:
        return self._choice_ids[:n]

    def _resolve_choice_ids(self) -> list[int]:
        ids = []
        for letter in LETTERS:
            s = self.style.letter_prefix + letter
            enc = self.tok.encode(s, add_special_tokens=False)
            if len(enc) != 1:
                raise ValueError(f"readout token {s!r} is not a single token: {enc}")
            ids.append(enc[0])
        if len(set(ids)) != len(ids):
            raise ValueError("choice letters map to duplicate token ids")
        return ids


@lru_cache(maxsize=1)
def default_style() -> PromptStyle:
    return PromptStyle()
