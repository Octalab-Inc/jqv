"""Branch isolation: changing question i must not change question j's distribution."""

from jqv.engine import make_engine
from jqv.types import Question

STATE = "橋梁Xの点検結果。主桁に腐食あり。床版は健全。"
Q_TARGET = Question(question="管理番号は何か。", choices=["1234", "5678", "9012"])


def _secret_probs(engine, rt):
    eng = make_engine(engine, rt)
    sibling_leak = Question(question="管理番号は 5678 です。この橋の床版は健全か。", choices=["健全", "損傷あり"])
    p_leak = eng.decide(STATE, [sibling_leak, Q_TARGET])[1].probabilities
    p_alone = eng.decide(STATE, [Q_TARGET])[0].probabilities
    p_state = eng.decide(STATE + " 管理番号は 5678。", [Q_TARGET])[0].probabilities
    return p_leak, p_alone, p_state


def test_packed_is_isolated(rt):
    p_leak, p_alone, p_state = _secret_probs("packed", rt)
    assert max(abs(a - b) for a, b in zip(p_leak, p_alone)) < 1e-3
    assert p_state[1] > 0.5  # secret in the shared state is visible


def test_packed_causal_leaks(rt):
    """Negative control: plain causal packing lets the sibling's secret leak into the target."""
    p_leak, p_alone, _ = _secret_probs("packed_causal", rt)
    assert p_leak[1] - p_alone[1] > 0.1
