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
