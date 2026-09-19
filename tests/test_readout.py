import math

from jqv.engine import make_engine
from jqv.types import Question


def test_probabilities_sum_to_one_and_confidence_bounds(rt, bridge_items):
    eng = make_engine("naive", rt)
    it = bridge_items[0]
    d = eng.decide(it["state"], [Question(question=it["question"], choices=it["choices"])])[0]
    assert math.isclose(sum(d.probabilities), 1.0, abs_tol=1e-5)
    assert len(d.probabilities) == len(it["choices"])
    assert 0.0 <= d.confidence <= 1.0
    assert 0.0 <= d.choice_mass <= 1.0
    assert d.argmax == it["answer"]


def test_temperature_changes_calibrated_only(rt, bridge_items):
    it = bridge_items[0]
    q = [Question(question=it["question"], choices=it["choices"])]
    d0 = make_engine("naive", rt).decide(it["state"], q)[0]
    d1 = make_engine("naive", rt, temperature=5.0).decide(it["state"], q)[0]
    assert d0.calibrated_probabilities is None
    assert d1.calibrated_probabilities is not None
    assert max(d1.calibrated_probabilities) < max(d1.probabilities) or max(d1.probabilities) == 1.0
