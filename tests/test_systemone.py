import math

from jqv.systemone import build_question, decide_systemone, format_answer
from jqv.types import Decision


class FakeRT:
    class prompt:
        @staticmethod
        def prefix_ids(state):
            return [0] * len(state.split())

        @staticmethod
        def suffix_ids(q, choices):
            return [0] * (len(q.split()) + len(choices))


class FakeEngine:
    rt = FakeRT()
    calibrated = False

    def decide(self, state, questions):
        out = []
        for q in questions:
            k = len(q.choices)
            p = [i + 1 for i in range(k)]
            s = sum(p)
            p = [x / s for x in p]
            cal = [1.0 / k] * k if self.calibrated else None
            out.append(Decision(logits=[0.0] * k, probabilities=p, calibrated_probabilities=cal, confidence=0.0))
        return out


def test_served_probabilities_are_calibrated_when_available():
    eng = FakeEngine()
    eng.calibrated = True
    body = {"state": "s", "questions": {"d": {"type": "choice", "instructions": "Pick.", "criteria": {"a": "", "b": ""}}}}
    out = decide_systemone(eng, body, "m")
    assert out["answers"]["d"]["probabilities"] == {"a": 0.5, "b": 0.5}


def test_choice_noul_score_mapping():
    q, meta = build_question("d", {"type": "choice", "instructions": "Pick.", "criteria": {"a": "first", "b": "second", "c": ""}})
    assert q.choices == ["a: first", "b: second", "c"] and meta["labels"] == ["a", "b", "c"]
    q, meta = build_question("d", {"type": "noul", "instructions": "Ok?", "criteria": {"true": "yes desc", "false": "no desc"}})
    assert q.choices == ["no: no desc", "yes: yes desc"] and meta["labels"] == ["no", "yes"]
    q, meta = build_question("d", {"type": "score", "instructions": "How many?", "criteria": ["none", "one", "two+"]})
    assert q.choices == ["0: none", "1: one", "2: two+"] and meta["labels"] == ["0", "1", "2"]


def test_answers_follow_the_typesafe_shapes():
    body = {"state": "some state text", "model": "jqv-test", "questions": {
        "c": {"type": "choice", "instructions": "Pick.", "criteria": {"a": "x", "b": "y", "c": "z"}},
        "n": {"type": "noul", "instructions": "Ok?", "criteria": {"true": "t", "false": "f"}},
        "s": {"type": "score", "instructions": "Count.", "criteria": ["0", "1", "2", "3"]}}}
    out = decide_systemone(FakeEngine(), body, "fallback")
    c, n, s = out["answers"]["c"], out["answers"]["n"], out["answers"]["s"]
    assert c["type"] == "choice" and c["choice"] == "c" and math.isclose(sum(c["probabilities"].values()), 1.0)
    assert set(c["probabilities"]) == {"a", "b", "c"}
    assert n["type"] == "noul" and math.isclose(n["noul"], 2 / 3)  # fake engine: p(no)=1/3, p(yes)=2/3
    assert s["type"] == "score" and list(s["probabilities"]) == ["0", "1", "2", "3"] and 0 <= s["score"] <= 3
    assert out["model"] == "jqv-test" and out["usage"]["input_tokens"] > 0 and out["usage"]["output_tokens"] == 0


def test_json_state_is_rendered():
    body = {"state": {"policy": ["a", "b"], "n": 1}, "questions": {"d": {"type": "noul", "instructions": "Ok?", "criteria": None}}}
    out = decide_systemone(FakeEngine(), body, "m")
    assert out["answers"]["d"]["type"] == "noul"
    import pytest
    with pytest.raises(Exception):
        decide_systemone(FakeEngine(), {"state": 3, "questions": {"d": {"type": "noul", "instructions": "x"}}}, "m")


def test_format_answer_argmax():
    a = format_answer({"type": "choice", "labels": ["x", "y"]}, [0.3, 0.7])
    assert a["choice"] == "y"


def test_real_engine_end_to_end(rt, bridge_items):
    from jqv.engine import make_engine

    it = bridge_items[0]
    body = {"state": it["state"], "model": "jqv", "questions": {
        "decision": {"type": "choice", "instructions": it["question"], "criteria": {c: "" for c in it["choices"]}}}}
    out = decide_systemone(make_engine("packed", rt), body, rt.model_id)
    ans = out["answers"]["decision"]
    assert ans["choice"] == it["choices"][it["answer"]]
    assert math.isclose(sum(ans["probabilities"].values()), 1.0, abs_tol=1e-5)


def test_query_first_layout_serves(rt, bridge_items):
    import dataclasses

    from jqv.engine import make_engine
    from jqv.prompt import PromptBuilder, PromptStyle
    from jqv.systemone import decide_systemone

    rt2 = dataclasses.replace(rt, prompt=PromptBuilder(rt.tokenizer, PromptStyle(layout="query_first")))
    it = bridge_items[0]
    body = {"state": it["state"], "questions": {"q": {"type": "choice", "instructions": it["question"],
                                                       "criteria": {f"opt_{i}": c for i, c in enumerate(it["choices"])}}}}
    out = decide_systemone(make_engine("naive", rt2), body, "test")
    assert set(out["answers"]["q"]["probabilities"]) == {f"opt_{i}" for i in range(len(it["choices"]))}
    assert out["usage"]["input_tokens"] > len(rt2.prompt.prefix_ids(it["state"], it["question"], it["choices"]))
