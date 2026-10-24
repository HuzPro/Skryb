from skryb.transcription.streaming.local_agreement import (
    LocalAgreementPolicy,
)


class TestLocalAgreementPolicy:
    def test_should_commit_nothing_on_first_hypothesis(self):
        policy = LocalAgreementPolicy()

        decision = policy.observe(["testing", "one"])

        assert decision.newly_committed == []
        assert decision.tentative == ["testing", "one"]
        assert policy.committed_words == []

    def test_should_commit_prefix_agreed_by_two_hypotheses(self):
        policy = LocalAgreementPolicy()
        policy.observe(["testing", "one", "two"])

        decision = policy.observe(["testing", "one", "three"])

        assert decision.newly_committed == ["testing", "one"]
        assert decision.tentative == ["three"]
        assert policy.committed_words == ["testing", "one"]

    def test_should_commit_incrementally_as_hypotheses_grow(self):
        policy = LocalAgreementPolicy()
        policy.observe(["one"])
        policy.observe(["one", "two"])

        decision = policy.observe(["one", "two", "three"])

        assert policy.committed_words == ["one", "two"]
        assert decision.tentative == ["three"]

    def test_should_keep_divergent_tail_tentative(self):
        policy = LocalAgreementPolicy()
        policy.observe(["the", "quick", "brown"])

        decision = policy.observe(["the", "slow", "green"])

        assert decision.newly_committed == ["the"]
        assert decision.tentative == ["slow", "green"]

    def test_should_not_commit_unstable_word_until_spelling_settles(self):
        policy = LocalAgreementPolicy()
        policy.observe(["please", "recogni"])

        decision = policy.observe(["please", "recognize", "speech"])

        assert decision.newly_committed == ["please"]
        assert "recognize" not in policy.committed_words

    def test_should_never_retract_committed_words(self):
        policy = LocalAgreementPolicy()
        policy.observe(["one", "two"])
        policy.observe(["one", "two"])  # commits both

        # A later, shorter/contradicting hypothesis must not un-commit.
        policy.observe(["one"])

        assert policy.committed_words == ["one", "two"]

    def test_flush_commits_the_remaining_words(self):
        policy = LocalAgreementPolicy()
        policy.observe(["one", "two", "x"])
        policy.observe(["one", "two", "three"])  # commits one, two

        decision = policy.flush(["one", "two", "three", "four"])

        assert decision.newly_committed == ["three", "four"]
        assert decision.tentative == []
        assert policy.committed_words == ["one", "two", "three", "four"]
