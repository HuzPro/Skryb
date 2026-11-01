"""Drafts for speculative decoding: another transcript of the same audio, followed token by token.

The decoder checks a batch of drafted tokens in one pass and keeps the ones it would have written anyway, so
the draft only decides how many tokens a pass yields, never which. When the decoder's output leaves the draft
(a comma where the draft had none, a word spelled differently) the draft picks up again where the output's
last few tokens appear in it, nearest first.
"""


class Draft:
    """Follows *tokens*, another transcript's token ids, as the real output grows."""

    def __init__(self, tokens: list[int]) -> None:
        self._tokens = tokens
        self._cursor = 0
        self._output: list[int] = []

    def propose(self, limit: int) -> list[int]:
        """Up to *limit* tokens the draft expects next."""
        return self._tokens[self._cursor:self._cursor + limit]

    def advance(self, tokens: list[int]) -> None:
        """Take in tokens the decoder has written."""
        for token in tokens:
            self._output.append(token)
            self._cursor = self._next_cursor(token)

    def _next_cursor(self, token: int) -> int:
        cursor = self._cursor
        if cursor < len(self._tokens) and self._tokens[cursor] == token:
            return cursor + 1
        for n in (3, 2, 1):
            tail = self._output[-n:]
            if len(tail) < n:
                continue
            found = [i for i in range(max(0, cursor - 4), min(len(self._tokens) - n, cursor + 8) + 1)
                     if self._tokens[i:i + n] == tail]
            if found:
                return min(found, key=lambda i: abs(i + n - 1 - cursor)) + n
        # Most often one word came out differently: carry on after the draft's version of it.
        return min(cursor + 1, len(self._tokens))
