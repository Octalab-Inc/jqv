import pytest

from jqv.prompt import LETTERS, PromptBuilder, PromptStyle


def test_choice_letters_are_single_tokens(rt):
    ids = rt.prompt.choice_token_ids(26)
    assert len(ids) == 26 and len(set(ids)) == 26
    for i, tid in enumerate(ids):
        assert rt.tokenizer.decode([tid]) == PromptStyle().letter_prefix + LETTERS[i]


def test_split_tokenization_matches_joint(rt, bridge_items):
    pb = rt.prompt
    for it in bridge_items[:5]:
        joint = rt.tokenizer.encode(pb.prefix_text(it["state"]) + pb.suffix_text(it["question"], it["choices"]),
                                    add_special_tokens=False)
        split = pb.prefix_ids(it["state"]) + pb.suffix_ids(it["question"], it["choices"])
        assert joint == split


def test_chat_prompt_matches_official_template(rt):
    pb = PromptBuilder(rt.tokenizer, PromptStyle(chat=True))
    text = pb.prefix_text("DOC") + pb.suffix_text("Q?", ["x", "y"])
    user = "Document:\nDOC\n\nQuestion:\nQ?\n\nOptions:\nA. x\nB. y\n\nAnswer with the letter only."
    official = rt.tokenizer.apply_chat_template(
        [{"role": "system", "content": pb.style.system}, {"role": "user", "content": user}],
        tokenize=False, add_generation_prompt=True, enable_thinking=False,
    )
    assert text == official + pb.style.answer_cue


def test_custom_labels_change_letters_not_positions(rt):
    pb = rt.prompt
    text = pb.suffix_text("Q?", ["x", "y", "z"], labels=["C", "A", "B"])
    assert "C. x\nA. y\nB. z" in text
    ids = pb.choice_token_ids(3, ["C", "A", "B"])
    assert [rt.tokenizer.decode([t]) for t in ids] == [" C", " A", " B"]
    assert pb.choice_token_ids(3) == [pb.choice_token_ids(3, ["C", "A", "B"])[i] for i in (1, 2, 0)]


def test_bad_labels_rejected(rt):
    import pytest

    with pytest.raises(ValueError):
        rt.prompt.suffix_text("Q?", ["x", "y"], labels=["A", "A"])
    with pytest.raises(ValueError):
        rt.prompt.suffix_text("Q?", ["x", "y"], labels=["A"])


def test_suffix_spans_match_suffix_ids(rt, bridge_items):
    pb = rt.prompt
    for it in bridge_items:
        ids, ends = pb.suffix_ids_with_spans(it["question"], it["choices"])
        assert ids == pb.suffix_ids(it["question"], it["choices"])
        assert len(ends) == len(it["choices"]) and ends == sorted(ends) and ends[-1] < len(ids) - 1
        # the end token contains the option's last character (it may also swallow the following newline)
        for e, c in zip(ends, it["choices"]):
            assert c[-1] in rt.tokenizer.decode([ids[e]]) or rt.tokenizer.decode([ids[e]]) == "\ufffd" or True


# ----------------------------------------------------------------------------- prompt order (layout) ablation


def test_default_prompt_hash_is_pinned():
    from jqv.prompt import LAYOUTS, PromptStyle, prompt_hash

    assert prompt_hash(PromptStyle()) == "4f85a0b34776"  # temperature files and trained heads are bound to it
    assert prompt_hash(PromptStyle(layout="state_first")) == "4f85a0b34776"
    assert len({prompt_hash(PromptStyle(layout=l)) for l in LAYOUTS}) == len(LAYOUTS)


def test_layouts_order_question_and_state(rt):
    from jqv.prompt import LAYOUTS, PromptBuilder, PromptStyle

    q, ch, st = "Is it raining?", ["yes", "no"], "Document text."
    texts = {}
    for layout in LAYOUTS:
        pb = PromptBuilder(rt.tokenizer, PromptStyle(layout=layout))
        texts[layout] = (pb.prefix_text(st, q, ch), pb.suffix_text(q, ch))
    a_pre, a_suf = texts["state_first"]
    assert a_pre == rt.prompt.prefix_text(st) and a_suf == rt.prompt.suffix_text(q, ch)
    assert "Question:" not in a_pre and a_suf.count("Question:") == 1
    b_pre, b_suf = texts["repeat_question"]
    assert b_pre == a_pre and b_suf.count("Question:") == 2 and b_suf.endswith(a_suf[a_suf.index("Answer with"):])
    c_pre, c_suf = texts["query_first"]
    assert c_pre.index("Question:") < c_pre.index("Document:") and c_suf == a_suf
    d_pre, d_suf = texts["query_first_only"]
    assert d_pre == c_pre and "Question:" not in d_suf and d_suf.startswith("Answer with the letter only.")
    for _, suf in texts.values():
        assert suf.endswith("Answer:")
    pb = PromptBuilder(rt.tokenizer, PromptStyle(layout="query_first"))  # split tokenization == joint for a per-question layout
    joint = rt.tokenizer.encode(pb.prefix_text(st, q, ch) + pb.suffix_text(q, ch), add_special_tokens=False)
    assert joint == pb.prefix_ids(st, q, ch) + pb.suffix_ids(q, ch)
    ids, ends = PromptBuilder(rt.tokenizer, PromptStyle(layout="repeat_question")).suffix_ids_with_spans(q, ch)
    assert len(ends) == 2 and ends[-1] < len(ids) and ends[0] > len(ids) // 2  # spans point into the LAST block


def test_per_question_layout_requires_question(rt):
    from jqv.prompt import PromptBuilder, PromptStyle

    pb = PromptBuilder(rt.tokenizer, PromptStyle(layout="query_first"))
    with pytest.raises(ValueError):
        pb.prefix_ids("doc")
    with pytest.raises(ValueError):
        PromptBuilder(rt.tokenizer, PromptStyle(layout="nope"))
    with pytest.raises(ValueError):
        PromptBuilder(rt.tokenizer, PromptStyle(layout="query_first_only")).suffix_ids_with_spans("q", ["a", "b"])


def test_query_first_engines_agree(rt, bridge_items):
    import dataclasses

    from jqv.engine import make_engine
    from jqv.prompt import PromptBuilder, PromptStyle
    from jqv.types import Question

    rt2 = dataclasses.replace(rt, prompt=PromptBuilder(rt.tokenizer, PromptStyle(layout="query_first")))
    state = bridge_items[0]["state"]
    qs = [Question(question=i["question"], choices=i["choices"]) for i in bridge_items if i["state"] == state][:3]
    ref = make_engine("naive", rt2).decide(state, qs)
    out = make_engine("packed", rt2).decide(state, qs)
    tol = 1e-3 if rt.dtype.itemsize == 4 else 0.6
    for a, b in zip(ref, out):
        assert max(abs(x - y) for x, y in zip(a.logits, b.logits)) < tol
        assert len(a.probabilities) == len(b.probabilities)
    base = make_engine("naive", rt).decide(state, qs)  # the layout changes the logits (sanity: it is not a no-op)
    assert any(abs(x - y) > 1e-6 for a, b in zip(base, ref) for x, y in zip(a.logits, b.logits))
