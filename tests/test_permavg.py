import math

from jqv.engine import make_engine
from jqv.fewshot import HEADER, fewshot_state, fixed_state, format_exemplar
from jqv.types import Question


def test_perm_avg_probabilities_sum_to_one_and_are_order_invariant(rt, bridge_items):
    it = bridge_items[0]
    eng = make_engine("packed", rt, perm_avg=True)
    q = Question(question=it["question"], choices=it["choices"])
    d = eng.decide(it["state"], [q])[0]
    assert math.isclose(sum(d.probabilities), 1.0, abs_tol=1e-5)
    assert d.perm_avg_k == len(it["choices"])
    # cyclic shifts of the input order give the same averaged distribution (up to bf16/fp32 noise), re-aligned
    k = len(it["choices"])
    q2 = Question(question=it["question"], choices=[it["choices"][(i + 1) % k] for i in range(k)])
    d2 = eng.decide(it["state"], [q2])[0]
    realigned = [d2.probabilities[(j - 1) % k] for j in range(k)]
    assert max(abs(a - b) for a, b in zip(d.probabilities, realigned)) < 1e-3


def test_fewshot_state_format():
    ex = format_exemplar({"question": "Q?", "choices": ["x", "y", "z"], "answer": 2})
    assert ex.endswith("Answer: C") and "A. x\nB. y\nC. z" in ex
    st = fewshot_state("abstract_algebra", 5)
    assert st.startswith(HEADER) and st.count("Answer: ") == 5
    assert fewshot_state("no_such_subject", 5) == fixed_state(5)
    assert fixed_state(5).count("Answer: ") == 5
