"""
Trains the language identifier and saves it for pinoybot.py.

Run from the project root: python3 -m train.train
"""

import pickle

import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from features import get_features

SEED = 42

data = pd.read_csv("train/labeled_dataset.csv")

# split by sentence (not by word) so words of one sentence never end up in two sets: 70/15/15
sentence_ids = data["sentence_id"].unique()
train_ids, rest_ids = train_test_split(sentence_ids, test_size=0.3, random_state=SEED)
val_ids, test_ids = train_test_split(rest_ids, test_size=0.5, random_state=SEED)


def build(ids):
    X = []
    y = []
    # groups by sentence and use get_features to describe each word
    for sentence_id, sentence in data[data["sentence_id"].isin(ids)].groupby("sentence_id", sort=False):
        sentence = sentence.sort_values("word_id")
        tokens = sentence["word"].tolist()
        for i in range(len(tokens)):
            X.append(get_features(tokens, i))
            y.append(sentence.iloc[i]["label"])
    return X, y


X_train, y_train = build(train_ids)
X_val, y_val = build(val_ids)
X_test, y_test = build(test_ids)
print(f"Sentences: {len(train_ids)} train, {len(val_ids)} validation, {len(test_ids)} test")

model = DecisionTreeClassifier(random_state=SEED)
model.fit(X_train, y_train)

# validation: use this score to compare features and settings
prediction = model.predict(X_val)
print("Validation accuracy:", accuracy_score(y_val, prediction))
print(classification_report(y_val, prediction))

# test: final score for the report, look at it only at the end
prediction = model.predict(X_test)
print("Test accuracy:", accuracy_score(y_test, prediction))
print(classification_report(y_test, prediction))

with open("pinoybot_model.pkl", "wb") as f:
    pickle.dump(model, f)
print("Model saved to pinoybot_model.pkl")
