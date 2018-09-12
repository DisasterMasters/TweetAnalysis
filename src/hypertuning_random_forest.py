import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from tqdm import tqdm

train = pd.read_csv('/home/manny/PycharmProjects/TweetAnalysis/florida/training_data/supervised_data/utility/utility_supervised_rf_8.15.2018_MC.csv')
train2 = pd.read_csv('/home/manny/PycharmProjects/TweetAnalysis/florida/training_data/supervised_data/utility/utility_supervised_new coding_1073.csv')


X = train['Tweet']
y = train['Manual Coding']
X2 = train2['Tweet']
y2 = train2['Code']

X = X.append(X2,  ignore_index=True)
y = y.append(y2,  ignore_index=True)



sum = 0
# for i in range(5):
#     X_train, X_test, y_train, y_test = train_test_split(X, y)
#
#     tfidf = TfidfVectorizer(sublinear_tf=True, min_df=5, norm='l2', encoding='latin-1', ngram_range=(1, 2), stop_words='english')
#     X_train_tfidf = tfidf.fit_transform(X_train)
#     X_test_tfidf = tfidf.transform(X_test)
#
#     model = RandomForestClassifier()
#     model.fit(X_train_tfidf, y_train)
#     score = model.score(X_test_tfidf, y_test)
#     sum += score
#     print("test score: " + str(score))
# print("=================" + "\n" +  "Average Score: " + str(sum /5))


# 5787644787644788
# 5776061776061777
# 5768339768339767

X_train, X_test, y_train, y_test = train_test_split(X, y)

tfidf = TfidfVectorizer(sublinear_tf=True, min_df=5, norm='l2', encoding='latin-1', ngram_range=(1, 2),
                        stop_words='english')
X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)
import matplotlib.pyplot as plt

from sklearn.grid_search import GridSearchCV
from time import time
grid_times = {}
clf = RandomForestClassifier(random_state = 84)

# for number in tqdm(np.arange(2, 600, 50)):
#     param = np.arange(1, number, 10)
#     param_grid = {"n_estimators": param,
#                   "criterion": ["gini", "entropy"]}
#
#     grid_search = GridSearchCV(clf, param_grid=param_grid)
#
#     t0 = time()
#     grid_search.fit(X_train_tfidf, y_train)
#     compute_time = time() - t0
#     grid_times[len(grid_search.grid_scores_)] = time() - t0
#
# grid_times = pd.DataFrame.from_dict(grid_times, orient='index')

#
# final = pd.DataFrame.from_dict(grid_times)
# final = final.sort_index()
# plt.plot(final.index.values, final[0])
# plt.xlabel('Number of Parameter Permutations')
# plt.ylabel('Time (sec)')
# plt.title('Time vs. Number of Parameter Permutations of GridSearchCV')


# function takes a RF parameter and a ranger and produces a plot and dataframe of CV scores for parameter values
def evaluate_param(parameter, num_range, index):
    grid_search = GridSearchCV(clf, param_grid={parameter: num_range})
    grid_search.fit(X_train_tfidf, y_train)

    df = {}
    for i, score in enumerate(grid_search.grid_scores_):
        df[score[0][parameter]] = score[1]

    df = pd.DataFrame.from_dict(df, orient='index')
    df.reset_index(level=0, inplace=True)
    df = df.sort_values(by='index')

    plt.subplot(3, 2, index)
    plot = plt.plot(df['index'], df[0])
    plt.title(parameter)
    plt.show()
    return plot, df


# parameters and ranges to plot
param_grid = {"n_estimators": np.arange(140, 160, 1),
              "max_depth": np.arange(60, 70, 1),
              "min_samples_split": np.arange(5, 15, 1),
              "max_leaf_nodes": np.arange(250, 350, 1),}

index = 1
# plt.figure(figsize=(16, 12))
# for parameter, param_range in tqdm(dict.items(param_grid)):
#     evaluate_param(parameter, param_range, index)
#     index += 1


from operator import itemgetter

# Utility function to report best scores
def report(grid_scores, n_top):
    top_scores = sorted(grid_scores, key=itemgetter(1), reverse=True)[:n_top]
    for i, score in enumerate(top_scores):
        print("Model with rank: {0}".format(i + 1))
        print("Mean validation score: {0:.4f})".format(
              score.mean_validation_score,
              np.std(score.cv_validation_scores)))
        print("Parameters: {0}".format(score.parameters))
        print("")

# parameters for GridSearchCV
param_grid2 = {"n_estimators": [150,151,152,153,154,155,156,157,],
              "max_depth": [60,61,62,63,64,65,66,67,68,69,70,],
              "min_samples_split": [6,7,8,9,10],
              "min_samples_leaf": [1,2,3,4, 5,],
              "max_leaf_nodes": [277, 278, 279, 280, 281, 282, 283, 284, 285 ],
              "min_weight_fraction_leaf": [0.1]}


grid_search = GridSearchCV(clf, param_grid=param_grid2, verbose=2)
grid_search.fit(X_train_tfidf, y_train)

report(grid_search.grid_scores_, 10)

