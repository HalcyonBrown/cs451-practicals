"""
In this lab, we once again have a mandatory 'python' challenge.
Then we have a more open-ended Machine Learning 'see why' challenge.

This data is the "Is Wikipedia Literary" that I pitched.
You can contribute to science or get a sense of the data here: https://label.jjfoley.me/wiki
"""

import gzip, json
from shared import dataset_local_path, TODO
from dataclasses import dataclass
from typing import Dict, List
from sklearn.ensemble import RandomForestClassifier


"""
Problem 1: We have a copy of Wikipedia (I spared you the other 6 million pages).
It is separate from our labels we collected.
"""


@dataclass
class JustWikiPage:
    title: str
    wiki_id: str
    body: str


# Load our pages into this pages list.
pages: List[JustWikiPage] = []
with gzip.open(dataset_local_path("tiny-wiki.jsonl.gz"), "rt") as fp:
    for line in fp:
        entry = json.loads(line)
        pages.append(JustWikiPage(**entry))


@dataclass
class JustWikiLabel:
    wiki_id: str
    is_literary: bool


# Load our judgments/labels/truths/ys into this labels list:
labels: List[JustWikiLabel] = []
with open(dataset_local_path("tiny-wiki-labels.jsonl")) as fp:
    for line in fp:
        entry = json.loads(line)
        labels.append(
            JustWikiLabel(wiki_id=entry["wiki_id"], is_literary=entry["truth_value"])
        )


@dataclass
class JoinedWikiData:
    wiki_id: str
    is_literary: bool
    title: str
    body: str


print(len(pages), len(labels))
print(pages[0])
print(labels[0])


# "1. create a list of JoinedWikiData from the ``pages`` and ``labels`` lists.")
# ***Solution help from Prof Foley in class
# This challenge has some very short solutions, so it's more conceptual. If you're stuck after ~10-20 minutes of thinking, ask!
joined_data: Dict[str, JoinedWikiData] = {}
# First I need to organize the labels list by wiki_id
labels_by_id: Dict[str, JustWikiLabel] = {}
for label in labels:
    labels_by_id[label.wiki_id] = label

# Second I need to use the wiki_id to find the corresponding page in order to pair the two together
for page in pages:
    # handle the case where a page is missing a label
    if page.wiki_id not in labels_by_id:
        print("This page is missing its label: ", page.wiki_id)  # or use the title
        continue
    # locate the label for the current page using the page's wiki_id
    page_label = labels_by_id[page.wiki_id]
    # assemble the new row with all the correct variables
    final_row = JoinedWikiData(
        page.wiki_id, page_label.is_literary, page.title, page.body
    )
    # finally, add the final_row to joined_data for each page in pages where the dictionary key is the wiki_id
    joined_data[final_row.wiki_id] = final_row

############### Problem 1 ends here ###############

# Make sure it is solved correctly!
assert len(joined_data) == len(pages)
assert len(joined_data) == len(labels)
# Make sure it has *some* positive labels!
assert sum([1 for d in joined_data.values() if d.is_literary]) > 0
# Make sure it has *some* negative labels!
assert sum([1 for d in joined_data.values() if not d.is_literary]) > 0

# Construct our ML problem:
ys = []
examples = []
for wiki_data in joined_data.values():
    ys.append(wiki_data.is_literary)
    examples.append(wiki_data.body)

## We're actually going to split before converting to features now...
from sklearn.model_selection import train_test_split
import numpy as np

RANDOM_SEED = 1234

## split off train/validate (tv) pieces.
ex_tv, ex_test, y_tv, y_test = train_test_split(
    examples,
    ys,
    train_size=0.75,
    shuffle=True,
    random_state=RANDOM_SEED,
)
# split off train, validate from (tv) pieces.
ex_train, ex_vali, y_train, y_vali = train_test_split(
    ex_tv, y_tv, train_size=0.66, shuffle=True, random_state=RANDOM_SEED
)

## Convert to features, train simple model (TFIDF will be explained eventually.)
from sklearn.feature_extraction.text import TfidfVectorizer

# Only learn columns for words in the training data, to be fair.
word_to_column = TfidfVectorizer(
    strip_accents="unicode", lowercase=True, stop_words="english", max_df=0.5
)
word_to_column.fit(ex_train)

# Test words should surprise us, actually!
X_train = word_to_column.transform(ex_train)
X_vali = word_to_column.transform(ex_vali)
X_test = word_to_column.transform(ex_test)


print("Ready to Learn!")
from sklearn.linear_model import LogisticRegression, SGDClassifier, Perceptron
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import roc_auc_score

models = {
    "SGDClassifier": SGDClassifier(),
    "Perceptron": Perceptron(),
    "LogisticRegression": LogisticRegression(),
    "DTree": DecisionTreeClassifier(),
    "RandomForest": RandomForestClassifier(),
}

for name, m in models.items():
    m.fit(X_train, y_train)
    print("{}:".format(name))
    print("\tVali-Acc: {:.3}".format(m.score(X_vali, y_vali)))
    if hasattr(m, "decision_function"):
        scores = m.decision_function(X_vali)
    else:
        scores = m.predict_proba(X_vali)[:, 1]
    print("\tVali-AUC: {:.3}".format(roc_auc_score(y_score=scores, y_true=y_vali)))

"""
Results should be something like:

SGDClassifier:
        Vali-Acc: 0.84
        Vali-AUC: 0.879
Perceptron:
        Vali-Acc: 0.815
        Vali-AUC: 0.844
LogisticRegression:
        Vali-Acc: 0.788
        Vali-AUC: 0.88
DTree:
        Vali-Acc: 0.739
        Vali-AUC: 0.71
RandomForest:
        Vali-Acc: 0.819
        Vali-AUC: 0.883
"""
# ("2. Explore why DecisionTrees are not beating linear models. Answer one of:")
# ("2.A. Is it a bad depth?")
# ("2.B. Do Random Forests do better?")
print(
    """Answer to Question 2.B. Do Random Forests do better?:\nThe Random Forest model yields better results than the Decision Tree model when run on our 
Is This Wiki Page Literary dataset. The Random Forest model sees results on par/similar
to the other linear models. While the Decision Tree runs through the dataset and makes splitting decisions
until it is done creating categories, the Random Forest acts like a combination of many Decision Trees 
in that it makes many randomized decisions and then ultimately makes a decision based on the majority of 
previous decisions. Like we saw in a previous practical, this method of bootstrapping can increase the 
accuracy of decision tree models."""
)
# ("2.C. Is it randomness? Use simple_boxplot and bootstrap_auc/bootstrap_acc to see if the differences are meaningful!")
# ("2.D. Is it randomness? Control for random_state parameters!")