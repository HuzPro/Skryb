from skryb.transcription.qwen3_asr_adapter import new_token_budget


def test_the_token_budget_never_starves_a_short_clip() -> None:
    assert new_token_budget(0.2) >= 64


def test_the_token_budget_grows_with_the_clip_so_a_long_dictation_is_not_cut_short() -> None:
    assert new_token_budget(120) > 5 * 120  # about 3 words, 4 to 5 tokens, a second
    assert new_token_budget(30) < new_token_budget(31)
