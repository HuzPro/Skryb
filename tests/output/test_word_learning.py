import json
from dataclasses import replace
from pathlib import Path

import pytest

from skryb.core.json_config_adapter import JsonConfigAdapter
from skryb.output.dictionary_config import (
    DictionaryConfig,
    LiteralRule,
    load_dictionary,
    save_dictionary,
)
from skryb.output.word_learning import (
    MAX_CANDIDATES,
    WordCorrection,
    WordLearner,
    learns_words,
    word_corrections,
)


def _sent(
    pasted: str,
    final: str,
    seq: int,
    reliable: bool = True,
    flags: tuple[str, ...] = (),
) -> dict[str, object]:
    return {
        "event": "sent",
        "block": 1,
        "reason": "enter",
        "process": "slack.exe",
        "text": final,
        "reliable": reliable,
        "flags": list(flags),
        "key_count": 7,
        "spans": [{"seq": seq, "pasted_text": pasted, "final_text": final}],
    }


class Setup:
    def __init__(self, tmp_path: Path, enabled: bool = True) -> None:
        self.tmp_path = tmp_path
        self.dictionary = tmp_path / "dictionary.json"
        self.store = tmp_path / "learning.json"
        self.enabled = enabled
        self.rebuilds = 0

    def learner(self, session: str = "session-a") -> WordLearner:
        return WordLearner(
            self.store, self.dictionary, lambda: self.enabled, self._rebuilt, session,
        )

    def _rebuilt(self) -> None:
        self.rebuilds += 1

    def learned(self) -> list[tuple[tuple[str, ...], str]]:
        return [
            (rule.spoken, rule.replacement)
            for rule in load_dictionary(self.dictionary).rules
            if isinstance(rule, LiteralRule) and rule.learned
        ]


@pytest.fixture
def setup(tmp_path: Path) -> Setup:
    return Setup(tmp_path)


class TestLearningWords:
    def test_two_reliable_corrections_from_two_dictations_teach_a_name_and_survive_restart(
        self, setup,
    ):
        learner = setup.learner()
        learner.observe(_sent("Send it to Jon today.", "Send it to John today.", seq=1))
        assert setup.learned() == []

        learner.observe(_sent("Ask Jon, please.", "Ask John, please.", seq=2))

        assert setup.learned() == [(("Jon",), "John")]
        assert setup.rebuilds == 1
        setup.learner("session-b").observe(_sent("Hi Jon", "Hi John", seq=3))
        assert setup.learned() == [(("Jon",), "John")]

    def test_one_correction_is_remembered_across_a_restart(self, setup):
        setup.learner("session-a").observe(_sent("Call Aisha.", "Call Aïsha.", seq=1))

        setup.learner("session-b").observe(_sent("Thanks Aisha!", "Thanks Aïsha!", seq=1))

        assert setup.learned() == [(("Aisha",), "Aïsha")]

    def test_the_same_correction_repeated_in_one_dictation_does_not_teach(self, setup):
        learner = setup.learner()
        event = _sent("Jon and Jon", "John and John", seq=4)
        event["spans"] = [*event["spans"], {"seq": 4, "pasted_text": "Jon", "final_text": "John"}]

        learner.observe(event)
        learner.observe(_sent("Hi Jon", "Hi John", seq=4))

        assert setup.learned() == []

    @pytest.mark.parametrize(
        "event",
        [
            _sent("Hi Jon", "Hi John", seq=1, reliable=False),
            _sent("Hi Jon", "Hi John", seq=1, flags=("select_all",)),
            {"event": "paste", "seq": 1},
        ],
        ids=["unreliable", "flagged", "not-a-sent-event"],
    )
    def test_unreliable_or_flagged_edits_teach_nothing(self, setup, event):
        learner = setup.learner()

        learner.observe(event)
        learner.observe(_sent("Hi Jon", "Hi John", seq=2))

        assert setup.learned() == []

    @pytest.mark.parametrize(
        ("pasted", "final"),
        [
            ("Meet at 3 today", "Meet at 4 today"),
            ("I went to the shop", "I went into the shop"),
            ("the report is late", "the invoice is late"),
            ("send it now", "please forward everything tomorrow morning"),
        ],
        ids=["digits", "function-word", "rewrite", "long-rewrite"],
    )
    def test_digits_function_words_and_rewrites_teach_nothing(self, setup, pasted, final):
        learner = setup.learner()

        learner.observe(_sent(pasted, final, seq=1))
        learner.observe(_sent(pasted, final, seq=2))

        assert setup.learned() == []

    def test_a_manual_rule_blocks_learning_of_that_word(self, setup):
        manual = LiteralRule(spoken=("Jon",), replacement="Jonathan")
        save_dictionary(setup.dictionary, DictionaryConfig(rules=(manual,)))
        learner = setup.learner()

        learner.observe(_sent("Hi Jon", "Hi John", seq=1))
        learner.observe(_sent("Bye Jon", "Bye John", seq=2))

        assert load_dictionary(setup.dictionary).rules == (manual,)

    def test_forgetting_a_learned_word_blocks_relearning_it(self, setup):
        learner = setup.learner()
        learner.observe(_sent("Hi Jon", "Hi John", seq=1))
        learner.observe(_sent("Bye Jon", "Bye John", seq=2))
        save_dictionary(setup.dictionary, DictionaryConfig())
        learner.block([("Jon", "John")])

        learner.observe(_sent("Hey Jon", "Hey John", seq=3))
        learner.observe(_sent("Yo Jon", "Yo John", seq=4))
        setup.learner("after-restart").observe(_sent("Hi Jon", "Hi John", seq=5))

        assert setup.learned() == []

    def test_2000_candidate_corrections_keep_the_store_bounded(self, setup):
        learner = setup.learner()
        for batch in range(20):
            event = _sent("", "", seq=0)
            event["spans"] = [
                {"seq": index, "pasted_text": f"Meet {name} now", "final_text": f"Meet {name}x now"}
                for index in range(batch * 100, batch * 100 + 100)
                for name in ["Zar" + "".join(chr(ord("a") + int(d)) for d in f"{index:04d}")]
            ]
            learner.observe(event)

        restarted = setup.learner("session-b")
        restarted.observe(_sent("Hi Jon", "Hi John", seq=1))

        assert len(restarted._candidates) <= MAX_CANDIDATES
        assert setup.store.stat().st_size < 150_000
        assert setup.learned() == []

    def test_nothing_is_learned_while_the_switch_is_off(self, tmp_path):
        setup = Setup(tmp_path, enabled=False)
        learner = setup.learner()

        for seq in range(5):
            learner.observe(_sent("Hi Jon.", "hi John", seq=seq))

        assert setup.learned() == []
        assert not setup.store.exists()

    def test_a_corrupt_store_is_ignored_and_rebuilt(self, setup):
        setup.store.write_text("{not json", encoding="utf-8")
        learner = setup.learner()

        learner.observe(_sent("Hi Jon", "Hi John", seq=1))
        learner.observe(_sent("Bye Jon", "Bye John", seq=2))

        assert setup.learned() == [(("Jon",), "John")]

    def test_a_store_with_old_style_tallies_still_loads_and_keeps_its_blocks(self, setup):
        setup.store.write_text(json.dumps({
            "version": 1,
            "candidates": [{"spoken": "Jon", "written": "John", "sources": ["old:1"], "seen": 3}],
            "blocked": [["siobhan", "Siobhán"]],
            "style": {"final_stop": 4, "contractions": -2},
        }), encoding="utf-8")
        learner = setup.learner()

        learner.observe(_sent("Hi Jon", "Hi John", seq=1))
        learner.observe(_sent("Hi Siobhan", "Hi Siobhán", seq=2))
        learner.observe(_sent("Bye Siobhan", "Bye Siobhán", seq=3))

        assert setup.learned() == [(("Jon",), "John")]
        assert "style" not in json.loads(setup.store.read_text(encoding="utf-8"))

    def test_an_unreadable_dictionary_is_never_overwritten_by_learning(self, setup):
        setup.dictionary.write_text("{ broken", encoding="utf-8")
        learner = setup.learner()

        learner.observe(_sent("Hi Jon", "Hi John", seq=1))
        learner.observe(_sent("Bye Jon", "Bye John", seq=2))

        assert setup.dictionary.read_text(encoding="utf-8") == "{ broken"


class TestTheLearningSwitch:
    @pytest.mark.parametrize(
        ("saved", "expected"),
        [
            ({}, True),
            ({"dictionary": {"learn_words": False}}, False),
            ({"dictionary": {"learn_words": True}}, True),
            ({"writing": {"cleanup": "light", "learn_words": False, "learn_style": True}}, False),
            ({"writing": {"learn_words": True}}, True),
            ({"writing": {"learn_words": False}, "dictionary": {"learn_words": True}}, True),
        ],
        ids=[
            "fresh-config", "turned-off", "turned-on", "off-in-an-old-config",
            "on-in-an-old-config", "the-new-switch-wins",
        ],
    )
    def test_the_switch_is_read_from_the_dictionary_settings_or_an_older_config(
        self, tmp_path, saved, expected,
    ):
        path = tmp_path / "config.json"
        path.write_text(json.dumps(saved), encoding="utf-8")
        config = JsonConfigAdapter(path)
        config.load()

        assert learns_words(config) is expected


class TestWordCorrections:
    def test_finds_name_and_spelling_fixes_with_accents_apostrophes_and_hyphens(self):
        found = word_corrections(
            "Tell Aisha and Jon that O'brien's co-worker called.",
            "Tell Aïsha and John that O'Brien's co-worker called.",
        )

        assert found == [
            WordCorrection("Aisha", "Aïsha"),
            WordCorrection("Jon", "John"),
            WordCorrection("O'brien's", "O'Brien's"),
        ]

    def test_ignores_edits_of_more_than_three_words(self):
        assert word_corrections(
            "one two three four five", "uno dos tres cuatro cinco",
        ) == []


def test_a_learned_rule_round_trips_through_the_dictionary_file_and_plain_rules_stay_plain(
    tmp_path,
):
    path = tmp_path / "dictionary.json"
    rules = (
        LiteralRule(spoken=("Jon",), replacement="John", learned=True),
        LiteralRule(spoken=("vtt",), replacement="Skryb"),
    )

    save_dictionary(path, DictionaryConfig(rules=rules))

    assert load_dictionary(path).rules == rules
    assert '"learned"' in path.read_text(encoding="utf-8")
    assert path.read_text(encoding="utf-8").count('"learned"') == 1
    assert replace(rules[0], learned=False) != rules[0]
