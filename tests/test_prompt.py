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
