"""
Turns the 3 annotator columns of final_dataset.csv into a single label per word.

Rule: majority vote (the tag chosen by at least 2 of the 3 groups).
The 2 words where all 3 groups disagree are decided by hand below.

Run from the project root: python3 train/prepare_data.py
"""

import csv
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
INPUT = ROOT / "final_dataset.csv"
OUTPUT = ROOT / "train" / "labeled_dataset.csv"

# Words with no majority (3 different tags), decided by our group. Key: word_id.
MANUAL_LABELS = {
    "5278": "ENG",  # "pesos" (sentence 231): English spelling, Filipino would be "piso"
    "6376": "ENG",  # "solon" (sentence 286): English word meaning legislator
}

rows = []
with open(INPUT, newline="", encoding="utf-8") as f:
    for row in csv.DictReader(f):
        votes = [row["Group 1"], row["Group 2"], row["Group 3"]]
        tag, count = Counter(votes).most_common(1)[0]
        if row["word_id"] in MANUAL_LABELS:
            tag = MANUAL_LABELS[row["word_id"]]
        elif count < 2:
            raise ValueError(f"No majority for word_id {row['word_id']} ({row['word']}): {votes}")
        rows.append([row["sentence_id"], row["word_id"], row["word"], tag])

# Sanity checks
assert len(rows) == 7515, len(rows)
assert {r[3] for r in rows} == {"ENG", "FIL", "OTH"}

with open(OUTPUT, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["sentence_id", "word_id", "word", "label"])
    writer.writerows(rows)

print(f"Wrote {len(rows)} rows to {OUTPUT.relative_to(ROOT)}")
print(Counter(r[3] for r in rows))
