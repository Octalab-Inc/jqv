"""Prompt construction.

The prompt is split into a *prefix* (shared state) and a *suffix* (one question with its
choices, ending in the answer cue). Prefix and suffix are tokenized separately so the
same token ids can be used by every engine (naive concat, KV-cache branch, packed branch).
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from functools import lru_cache

LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
PROMPT_FORMAT_VERSION = 1  # bump when prefix_text / suffix_text layout changes

# Prompt-order ablation. state_first is jqv's design (the state is prefilled once and shared by every question);
# repeat_question keeps the shared prefix and puts the question block twice before the answer cue; query_first and
# query_first_only put the question BEFORE the state, so the state's representation is question-conditioned but the
# prefix is per question (no sharing). See docs/report.md, "Prompt order".
LAYOUTS = ("state_first", "repeat_question", "query_first", "query_first_only")
PER_QUESTION_PREFIX_LAYOUTS = ("query_first", "query_first_only")

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
    layout: str = "state_first"  # one of LAYOUTS


def prompt_hash(style: PromptStyle) -> str:
    """Short stable id of the prompt layout + style. Calibration is only valid for the prompt it was fitted on."""
    fields = asdict(style)
    if fields.get("layout", "state_first") == "state_first":
        fields.pop("layout", None)  # the default layout keeps the hash it had before `layout` existed (4f85a0b34776)
    payload = json.dumps({"format": PROMPT_FORMAT_VERSION, **fields}, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]


class PromptBuilder:
    def __init__(self, tokenizer, style: PromptStyle | None = None):
        self.tok = tokenizer
        self.style = style or PromptStyle()
        if self.style.layout not in LAYOUTS:
            raise ValueError(f"unknown prompt layout {self.style.layout!r}; choose from {LAYOUTS}")
        self._choice_ids = self._resolve_choice_ids()

    @property
    def hash(self) -> str:
        return prompt_hash(self.style)

    @property
    def per_question_prefix(self) -> bool:
        """True when the prefix contains the question (query-first layouts): nothing can be shared across questions."""
        return self.style.layout in PER_QUESTION_PREFIX_LAYOUTS

    # ----- text -----
    def _head(self) -> str:
        if self.style.chat:
            return f"<|im_start|>system\n{self.style.system}<|im_end|>\n<|im_start|>user\n"
        return ""

    @staticmethod
    def _doc(state: str) -> str:
        return f"Document:\n{state}\n\n" if state.strip() else ""

    def question_block(self, question: str, choices: list[str], labels: list[str] | None = None) -> str:
        labels = self.resolve_labels(len(choices), labels)
        opts = "\n".join(f"{labels[i]}. {c}" for i, c in enumerate(choices))
        return f"Question:\n{question}\n\nOptions:\n{opts}"

    def prefix_text(self, state: str, question: str | None = None, choices: list[str] | None = None,
                    labels: list[str] | None = None) -> str:
        if self.per_question_prefix:
            if question is None or choices is None:
                raise ValueError(f"layout {self.style.layout!r} puts the question before the state: pass question and choices")
            return self._head() + self.question_block(question, choices, labels) + "\n\n" + self._doc(state)
        return self._head() + self._doc(state)

    @staticmethod
    def resolve_labels(n: int, labels: list[str] | None) -> list[str]:
        """Letter printed in front of position i. Default A, B, C, ...; custom labels let experiments
        permute which letter each position carries (letter-token prior vs. position/order effects)."""
        if labels is None:
            return list(LETTERS[:n])
        if len(labels) != n or len(set(labels)) != n or any(l not in LETTERS for l in labels):
            raise ValueError(f"labels must be {n} distinct letters from A-Z, got {labels}")
        return list(labels)

    def _tail(self) -> str:
        if self.style.chat:
            return f"<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n{self.style.answer_cue}"
        return f"\n{self.style.answer_cue}"

    def suffix_text(self, question: str, choices: list[str], labels: list[str] | None = None) -> str:
        if not 2 <= len(choices) <= len(LETTERS):
            raise ValueError(f"choices must have 2..{len(LETTERS)} entries, got {len(choices)}")
        block = self.question_block(question, choices, labels)
        instr = "Answer with the letter only."
        layout = self.style.layout
        if layout == "repeat_question":
            body = f"{block}\n\n{block}\n\n{instr}"
        elif layout == "query_first_only":
            body = instr
        else:  # state_first, query_first
            body = f"{block}\n\n{instr}"
        return body + self._tail()

    # ----- ids -----
    def prefix_ids(self, state: str, question: str | None = None, choices: list[str] | None = None,
                   labels: list[str] | None = None) -> list[int]:
        return self.tok.encode(self.prefix_text(state, question, choices, labels), add_special_tokens=False)

    def suffix_ids(self, question: str, choices: list[str], labels: list[str] | None = None) -> list[int]:
        return self.tok.encode(self.suffix_text(question, choices, labels), add_special_tokens=False)

    def suffix_ids_with_spans(self, question: str, choices: list[str], labels: list[str] | None = None
                              ) -> tuple[list[int], list[int]]:
        """Suffix token ids (identical to `suffix_ids`) plus, for each option, the index of the token that
        contains the last character of the option text. Uses the fast tokenizer's offset mapping, so merged
        tokens such as "）\n\n" (CJK punctuation + newlines) are handled: that token still ends the option."""
        labels = self.resolve_labels(len(choices), labels)
        text = self.suffix_text(question, choices, labels)
        enc = self.tok(text, add_special_tokens=False, return_offsets_mapping=True)
        ids, offsets = enc["input_ids"], enc["offset_mapping"]
        head = f"Question:\n{question}\n\nOptions:\n"
        start = text.rfind(self.question_block(question, choices, labels))  # the LAST question block in the suffix
        if start < 0:
            raise ValueError(f"layout {self.style.layout!r} has no options in the suffix; the slot head needs option spans there")
        pos, ends = start + len(head), []
        for i, c in enumerate(choices):
            line = f"{labels[i]}. {c}"
            char_end = pos + len(line) - 1  # index of the option's last character
            ends.append(next(t for t, (a, b) in enumerate(offsets) if a <= char_end < b))
            pos += len(line) + 1  # "\n"
        return ids, ends

    def choice_token_ids(self, n: int, labels: list[str] | None = None) -> list[int]:
        """Readout token id for position i (= the letter printed at position i)."""
        return [self._choice_ids[LETTERS.index(l)] for l in self.resolve_labels(n, labels)]

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
