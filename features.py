"""
Feature extraction shared by train/train.py and pinoybot.py.

get_features(tokens, i) turns the word at position i of a sentence into a list of numbers.
The list must always have the same length and order. New features go at the end of the list.
"""

from typing import List

VOWELS = set("aeiouAEIOU")


def get_features(tokens: List[str], i: int) -> List[float]:
    word = tokens[i]
    length = len(word)
    return [
        length,                                                   # 1. word length
        int(word[:1].isupper()),                                  # 2. starts with a capital letter
        int(word.isalpha()),                                      # 3. only letters
        sum(c in VOWELS for c in word) / max(length, 1),          # 4. vowel ratio
    ]


if __name__ == "__main__":
    tokens = ["Love", "kita", ".", "DLSU"]
    assert get_features(tokens, 0) == [4, 1, 1, 0.5]
    assert get_features(tokens, 1) == [4, 0, 1, 0.5]
    assert get_features(tokens, 2) == [1, 0, 0, 0.0]
    assert get_features(tokens, 3) == [4, 1, 1, 0.25]
    print("features.py OK")
