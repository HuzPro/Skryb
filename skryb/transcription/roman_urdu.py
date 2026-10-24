"""Hindi script (Devanagari) to Roman Urdu, the way Urdu is usually typed in Latin letters.

Qwen3-ASR has no Urdu: it writes spoken Urdu as Hindi, in a script most Urdu readers can't read.
"""
import re
import unicodedata

_CONSONANTS = {
    "क": "k", "ख": "kh", "ग": "g", "घ": "gh", "ङ": "n", "च": "ch", "छ": "chh", "ज": "j", "झ": "jh", "ञ": "n",
    "ट": "t", "ठ": "th", "ड": "d", "ढ": "dh", "ण": "n", "त": "t", "थ": "th", "द": "d", "ध": "dh", "न": "n",
    "प": "p", "फ": "ph", "ब": "b", "भ": "bh", "म": "m", "य": "y", "र": "r", "ल": "l", "व": "w", "श": "sh",
    "ष": "sh", "स": "s", "ह": "h", "ळ": "l",
}
_NUKTA_FORMS = {"क": "q", "ख": "kh", "ग": "gh", "ज": "z", "झ": "zh", "ड": "r", "ढ": "rh", "फ": "f"}
_VOWELS = {
    "अ": "a", "आ": "aa", "इ": "i", "ई": "i", "उ": "u", "ऊ": "u", "ऋ": "ri", "ए": "e", "ऐ": "ai",
    "ओ": "o", "औ": "au", "ऑ": "o", "ऍ": "e",
}
_SIGNS = {
    "ा": "aa", "ि": "i", "ी": "i", "ु": "u", "ू": "u", "ृ": "ri", "े": "e", "ै": "ai", "ो": "o", "ौ": "au",
    "ॉ": "o", "ॅ": "e",
}
_NUKTA, _VIRAMA, _NASALS, _VISARGA = "़", "्", "ँं", "ः"
_OTHER = {"।": ".", "॥": ".", **{chr(0x0966 + d): str(d) for d in range(10)}}
_WORD = re.compile(r"[ऀ-ॿ]+")


def romanize(text: str) -> str:
    if not _WORD.search(text):
        return text
    return _WORD.sub(lambda m: _word(m.group()), unicodedata.normalize("NFC", text))


def _word(word: str) -> str:
    syllables = _syllables(word)  # [consonant, vowel, inherent, coda]
    _drop_silent_a(syllables)
    full = sum(1 for s in syllables if s[1])
    out = []
    for k, (consonant, vowel, _, coda) in enumerate(syllables):
        ends_word = k == len(syllables) - 1 and consonant and not coda
        if vowel == "aa" and (full > 1 or ends_word):
            vowel = "a"  # Roman Urdu writes "mera", "tha", "hamare" but "kaam", "aap", "aa"
        out.append(consonant + vowel + coda)
    return "".join(out)


def _syllables(word: str) -> list[list]:
    syllables, i = [], 0
    while i < len(word):
        ch = word[i]
        i += 1
        if ch in _CONSONANTS:
            consonant = _CONSONANTS[ch]
            if i < len(word) and word[i] == _NUKTA:
                consonant = _NUKTA_FORMS.get(ch, consonant)
                i += 1
            if i < len(word) and word[i] in _SIGNS:
                syllables.append([consonant, _SIGNS[word[i]], False, ""])
                i += 1
            elif i < len(word) and word[i] == _VIRAMA:
                syllables.append([consonant, "", False, ""])
                i += 1
            else:
                syllables.append([consonant, "a", True, ""])
        elif ch in _VOWELS:
            syllables.append(["", _VOWELS[ch], False, ""])
        elif ch in _NASALS or ch == _VISARGA:
            if syllables:
                syllables[-1][3] += "h" if ch == _VISARGA else "n"
        elif ch in _OTHER:
            syllables.append(["", "", False, _OTHER[ch]])
    return syllables


def _drop_silent_a(syllables: list[list]) -> None:
    """Hindi drops the written 'a' at the end of a word and between a vowel and
    a consonant+vowel: करते is "karte", not "karate"."""
    last = len(syllables) - 1
    if last > 0 and syllables[last][2] and not syllables[last][3]:
        syllables[last][1] = ""
    for k in range(last - 1, 0, -1):
        s = syllables[k]
        if s[2] and not s[3] and syllables[k - 1][1] and syllables[k + 1][0] and syllables[k + 1][1]:
            s[1] = ""
