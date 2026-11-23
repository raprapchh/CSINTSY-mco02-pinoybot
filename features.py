"""
Feature extraction shared by train/train.py and pinoybot.py.

get_features(tokens, i) turns the word at position i of a sentence into a list of numbers.
The list must always have the same length and order. New features go at the end of the list.
"""

from typing import List

VOWELS = set("aeiou")

# Common Tagalog verb/noun prefixes (morphology, not a word list)
FIL_PREFIXES = ("nag", "mag", "pag", "nak", "nam", "kan", "hin", "pin", "naka", "maka", "pina", "ipa")

# Letter pairs much more frequent in one language than the other
FIL_BIGRAMS = ("ng", "na", "sa", "ka", "ma", "pa", "ga", "ya", "ag", "ay")
ENG_BIGRAMS = ("th", "er", "re", "ed", "ou", "ve", "nt", "he", "or", "en")

# One column per ending
SUFFIXES = ("ng", "an", "in", "ed", "ty", "er", "al", "ly", "tion")


def get_features(tokens: List[str], i: int) -> List[float]:
    word = tokens[i]
    low = word.lower()
    length = len(word)
    bigrams = [low[k:k + 2] for k in range(length - 1)]
    return [
        length,                                                   # 1. word length
        int(word[:1].isupper()),                                  # 2. starts with a capital letter
        int(word.isalpha()),                                      # 3. only letters
        sum(c in VOWELS for c in low) / max(length, 1),           # 4. vowel ratio
        int(word[:1].isupper() and i > 0),                        # 5. capital not at sentence start (names)
        int(length == 1),                                         # 6. single character (initials, punctuation)
        int(low[-1:] in VOWELS),                                  # 7. ends with a vowel
        int(low.startswith(FIL_PREFIXES)),                        # 8. starts with a Tagalog prefix
        int("-" in word),                                         # 9. contains a hyphen (nag-march)
        sum(b in FIL_BIGRAMS for b in bigrams),                   # 10. count of Filipino-like letter pairs
        sum(b in ENG_BIGRAMS for b in bigrams),                   # 11. count of English-like letter pairs
        *[int(low.endswith(s)) for s in SUFFIXES],                # 12-20. one column per ending
    ]


if __name__ == "__main__":
    tokens = ["Love", "kita", ".", "DLSU", "nag-march", "Garcia"]
    f = [get_features(tokens, i) for i in range(len(tokens))]
    assert len({len(x) for x in f}) == 1, "all words must have the same number of features"
    assert f[0][:4] == [4, 1, 1, 0.5]
    assert f[1][:4] == [4, 0, 1, 0.5]
    assert f[2][:4] == [1, 0, 0, 0.0]
    assert f[3][:4] == [4, 1, 1, 0.25]
    assert f[0][4] == 0 and f[5][4] == 1        # capital at start vs in the middle
    assert f[2][5] == 1                          # "." is a single character
    assert f[4][7] == 1 and f[4][8] == 1         # nag-march: Tagalog prefix and hyphen
    print(f"features.py OK ({len(f[0])} features)")
