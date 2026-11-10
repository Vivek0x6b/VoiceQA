"""Word error rate: word-level edit distance divided by the reference length.

Text is normalised before comparing: case folded, punctuation dropped,
hyphens split words, apostrophes inside a word are kept ("don't"), and
numbers stay as written, so "20" and "twenty" count as different words.
"""

import re

# A number with inner separators (1,000 or 3.5), or a run of letters and
# digits with optional inner apostrophes.
_TOKEN = re.compile(r"\d+(?:[.,:]\d+)+|[^\W_]+(?:'[^\W_]+)*")
# Right and left single quotes and the modifier letter apostrophe.
_APOSTROPHES = str.maketrans({chr(0x2019): "'", chr(0x2018): "'", chr(0x02BC): "'"})


def normalize_words(text: str) -> list[str]:
    return _TOKEN.findall(text.translate(_APOSTROPHES).casefold())


def wer(reference: str, hypothesis: str) -> float:
    """Substitutions, deletions and insertions per reference word.

    Can be above 1.0 when the hypothesis has many extra words. An empty
    reference (after normalising) has no defined WER and raises ValueError.
    """
    ref = normalize_words(reference)
    hyp = normalize_words(hypothesis)
    if not ref:
        raise ValueError("WER needs a reference with at least one word")
    # One row of the Levenshtein table at a time: row[j] is the distance
    # between the reference words so far and the first j hypothesis words.
    row = list(range(len(hyp) + 1))
    for i, r in enumerate(ref, 1):
        diagonal, row[0] = row[0], i
        for j, h in enumerate(hyp, 1):
            diagonal, row[j] = row[j], min(row[j] + 1, row[j - 1] + 1, diagonal + (r != h))
    return row[-1] / len(ref)
