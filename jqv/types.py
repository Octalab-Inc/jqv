from __future__ import annotations

from pydantic import BaseModel, Field


class Question(BaseModel):
    question: str
    choices: list[str] = Field(min_length=2, max_length=26)


class Decision(BaseModel):
    """Result of one question.

    logits: raw next-token logits of the choice letters (length = len(choices)).
    probabilities: softmax(logits) (temperature 1).
    calibrated_probabilities: softmax(logits / T) with fitted temperature, or None.
    confidence: Jev-compatible (p_max - 1/K) / (1 - 1/K), i.e. distance from uniform (post-hoc; not a probability).
    entropy_concentration: 1 - H(p)/log(K), an entropy-based alternative summary (post-hoc; not a probability).
    choice_mass: total probability mass the full-vocab softmax puts on the choice letters (diagnostic).
    """

    logits: list[float]
    probabilities: list[float]
    calibrated_probabilities: list[float] | None = None
    confidence: float
    entropy_concentration: float | None = None
    choice_mass: float | None = None
    parsed: bool | None = None  # generate engine only: whether a letter could be parsed

    @property
    def argmax(self) -> int:
        p = self.calibrated_probabilities or self.probabilities
        return max(range(len(p)), key=lambda i: p[i])


class DecisionRequest(BaseModel):
    state: str = ""
    questions: list[Question] = Field(min_length=1)


class CalibrationInfo(BaseModel):
    """Provenance of the temperature used for `calibrated_probabilities`. Calibration is only valid for the
    distribution it was fitted on; consumers should check `dataset` before trusting the numbers elsewhere."""

    temperature: float
    model: str | None = None
    engine: str | None = None
    prompt_hash: str | None = None
    dataset: str | None = None
    n_val: int | None = None
    choice_counts: list[int] | None = None
    fitted_at: str | None = None
    dtype: str | None = None


class DecisionResponse(BaseModel):
    engine: str
    model: str
    prompt_hash: str
    temperature: float
    calibration: CalibrationInfo | None = None  # None when no temperature file is configured
    decisions: list[Decision]
