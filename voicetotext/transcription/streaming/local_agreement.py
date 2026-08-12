from dataclasses import dataclass


@dataclass(frozen=True)
class CommitDecision:
    """Outcome of feeding one hypothesis to :class:`LocalAgreementPolicy`."""

    newly_committed: list[str]
    tentative: list[str]


class LocalAgreementPolicy:
    """LocalAgreement-2 stabiliser (Strategy).

    A streaming decoder re-emits its whole hypothesis each tick, and the
    tail keeps changing as more audio arrives. This policy commits only
    the longest prefix that two consecutive hypotheses agree on - so
    committed text never has to be retracted - and reports the unstable
    remainder as tentative. It knows nothing about audio or models.
    """

    def __init__(self) -> None:
        self._prev_words: list[str] = []
        self._committed: list[str] = []

    def observe(self, hypothesis_words: list[str]) -> CommitDecision:
        agreed_len = _common_prefix_length(self._prev_words, hypothesis_words)
        newly_committed = hypothesis_words[len(self._committed):agreed_len]
        self._committed.extend(newly_committed)
        tentative = hypothesis_words[len(self._committed):]
        self._prev_words = list(hypothesis_words)
        return CommitDecision(
            newly_committed=newly_committed, tentative=tentative,
        )

    def flush(self, final_words: list[str]) -> CommitDecision:
        """Commit everything left in *final_words* - no more audio is coming."""
        newly_committed = final_words[len(self._committed):]
        self._committed.extend(newly_committed)
        self._prev_words = list(final_words)
        return CommitDecision(newly_committed=newly_committed, tentative=[])

    @property
    def committed_words(self) -> list[str]:
        return list(self._committed)


def _common_prefix_length(left: list[str], right: list[str]) -> int:
    length = 0
    for left_word, right_word in zip(left, right):
        if left_word != right_word:
            break
        length += 1
    return length
