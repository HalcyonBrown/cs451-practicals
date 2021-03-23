"""
Halcyon Brown
Practical 03
3/15/2021

In this lab, we'll go ahead and use the sklearn API to compare some real models over real data!
Same data as last time, for now.

Documentation:
https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html

We'll need to install sklearn, and numpy.
Use pip:

    # install everything from the requirements file.
    pip3 install -r requirements.txt
"""

#%%

# We won't be able to get past these import statments if you don't install the libraries!
# external libraries:
from sklearn.linear_model import Perceptron
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils import resample
from sklearn.metrics import accuracy_score
import numpy as np
import matplotlib.pyplot as plt

# standard python
import json
from dataclasses import dataclass
from typing import Dict, Any, List

# helper functions I made
from shared import dataset_local_path, TODO

#%% load up the data
examples = []
ys = []

with open(dataset_local_path("poetry_id.jsonl")) as fp:
    for line in fp:
        info = json.loads(line)
        # Note: the data contains a whole bunch of extra stuff; we just want numeric features for now.
        keep = info["features"]
        # whether or not it's poetry is our label.
        ys.append(info["poetry"])
        # hold onto this single dictionary.
        examples.append(keep)

#%% Convert data to 'matrices'
# We did this manually in p02, but SciKit-Learn has a tool for that:

from sklearn.feature_extraction import DictVectorizer

feature_numbering = DictVectorizer(sort=True)
feature_numbering.fit(examples)
X = feature_numbering.transform(examples)

print("Features as {} matrix.".format(X.shape))
#%% Set up our ML problem:

from sklearn.model_selection import train_test_split

RANDOM_SEED = 12345678

# Numpy-arrays are more useful than python's lists.
y = np.array(ys)
# split off train/validate (tv) pieces.
X_tv, X_test, y_tv, y_test = train_test_split(
    X, y, train_size=0.75, shuffle=True, random_state=RANDOM_SEED
)
# split off train, validate from (tv) pieces.
X_train, X_vali, y_train, y_vali = train_test_split(
    X_tv, y_tv, train_size=0.66, shuffle=True, random_state=RANDOM_SEED
)

# In this lab, we'll ignore test data, for the most part.

#%% DecisionTree Parameters:
params = {
    "criterion": "gini",
    "splitter": "best",  # changing this to random = much higher variance
    "max_depth": 5,
}

# train 100 different models, with different sources of randomness:
N_MODELS = 100
# sample 1 of them 100 times, to get a distribution of data for that!
N_SAMPLES = 100

# Calculating the seed-based accuracies
seed_based_accuracies = []
for randomness in range(N_MODELS):
    f_seed = DecisionTreeClassifier(random_state=RANDOM_SEED + randomness, **params)
    f_seed.fit(X_train, y_train)
    seed_based_accuracies.append(f_seed.score(X_vali, y_vali))

# Calculating the bootstrap-based accuracies
bootstrap_based_accuracies = []
# single seed, bootstrap-sampling of predictions:
f_single = DecisionTreeClassifier(random_state=RANDOM_SEED, **params)
f_single.fit(X_train, y_train)
y_pred = f_single.predict(X_vali)

# do the bootstrap:
for trial in range(N_SAMPLES):
    sample_pred, sample_truth = resample(
        y_pred, y_vali, random_state=trial + RANDOM_SEED
    )
    score = accuracy_score(y_true=sample_truth, y_pred=sample_pred)
    bootstrap_based_accuracies.append(score)

"""
# Creating the boxplot to visualize our results
boxplot_data: List[List[float]] = [seed_based_accuracies, bootstrap_based_accuracies]
plt.boxplot(boxplot_data)
plt.xticks(ticks=[1, 2], labels=["Seed-Based", "Bootstrap-Based"])
plt.xlabel("Sampling Method")
plt.ylabel("Accuracy")
plt.ylim([0.8, 1.0])
plt.show()
# if plt.show is not working, try opening the result of plt.savefig instead!
# plt.savefig("dtree-variance.png") # This doesn't work well on repl.it.
"""
# **************************************************************************
# Practical Questions
# **************************************************************************

# 1. understand/compare the bounds generated between the two methods.
print(
    """Answer to Question 1:\nWhen the splitter parameter is defined as best, there is less variation in the two 
different methods than when this parameter is set to random. The Seed-Based method when 
using the splitter parameter had minimal spread with y bounds ranging from about 0.906
to 0.915. The median looks to be about central within the IQR, with slightly more spread
along the lower whisker. In comparison, the Bootstrap-Based method, with the splitter parameter
set to best, resulted in more overall variance (y bounds from about 0.888 to 0.935), indicated 
by longer whiskers and a bigger IQR box. The median remains central. The Bootrapping method in 
this case resulted in visibly more variance in accuracy. This leads me to believe that 
decision trees are not greatly affected by randomness."""
)

# "2. Do one of the two following experiments."
# "2A. Evaluation++: what happens to the variance if we do K bootstrap samples for each of M models?"
# "2B. Return to experimenting on the decision tree: modify the plot to show ~10 max_depths of the decision tree."

# CODE FOR QUESTION 2B
max_depth = 11
N2_SAMPLES = 100
depth_based_accuracies = []

for depth in range(1, max_depth):
    # start with a single seed
    f_single = DecisionTreeClassifier(random_state=RANDOM_SEED, max_depth=depth)
    f_single.fit(X_train, y_train)
    y_pred = f_single.predict(X_vali)
    depth_specific_accuracies = []

    # perform and calculate accuracies from 100 bootstrap samples
    for trial in range(N2_SAMPLES):
        sample_pred, sample_truth = resample(
            y_pred, y_vali, random_state=trial + RANDOM_SEED
        )
        score = accuracy_score(y_true=sample_truth, y_pred=sample_pred)
        depth_specific_accuracies.append(score)
    depth_based_accuracies.append(depth_specific_accuracies)


# CREATING BOXPLOT FOR 2B
boxplot_data: List[List[float]] = depth_based_accuracies
plt.boxplot(boxplot_data)
plt.xticks(
    ticks=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    labels=["1", "2", "3", "4", "5", "6", "7", "8", "9", "10"],
)
plt.xlabel("Depth Level")
plt.ylabel("Bootstrap Method Accuracy")
plt.ylim([0.7, 1.0])
plt.show()