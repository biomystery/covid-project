import sys, os
import pandas as pd
import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.metrics import accuracy_score, confusion_matrix
import datetime

DATE = lambda: datetime.datetime.now().strftime("[%m/%d/%Y %H:%M:%S]")


def tune_elasticnet(X_train_, y_train_, rand_seed_, log_grid_, n_core=6):
    """tune elasticnet model one round"""
    log_estimator = LogisticRegression(solver='saga',
                                       penalty='elasticnet',
                                       random_state=rand_seed_)

    log_model = GridSearchCV(estimator=log_estimator,
                             param_grid=log_grid_,
                             verbose=0,
                             n_jobs=n_core,
                             cv=3,
                             scoring='accuracy')

    log_model.fit(X_train_, np.ravel(y_train_))

    return log_model


########################################################################
## read input 
########################################################################

# Import labels (for the whole dataset, both training and testing)
y = pd.read_csv('../data_processed/meta_data.csv', index_col=1)
print(y.shape)
y = y[[ 'Day Post-Infection']]

# merge adult and aged group
y['stage'] = [
    'early' if i in [2.0, 1.0, 0.5] else 'late'
    for i in y['Day Post-Infection']
]
# y['stage'].value_counts()

y = y.replace({'early': 0, 'late': 1})
labels = ['early', 'late']  # for plotting convenience later on
y[['Day Post-Infection', 'stage']].value_counts()


X = pd.read_csv('../data_processed/plot_dat.csv', index_col=0).T
print(X.shape)

X.head(1)
all(X.index == y.index)

########################################################################
## bootstrap: preparing 
########################################################################
# split train and testing
from sklearn.model_selection import train_test_split
import random

n_keep = 1
n_try = 1
test_idx_set_dict = {}
init_seed = 200
random.seed(a=init_seed, version=2)
while n_keep <= 100:
    rand_seed = random.randrange(1000, )
    print("try {0}: rand seed={1}, recorded sets={2}.".format(
        n_try, rand_seed, n_keep))
    X_train, X_test, y_train, y_test = train_test_split(X,
                                                        y,
                                                        test_size=0.2,
                                                        random_state=rand_seed,
                                                        stratify=y)
    if (set(y_test.index) not in test_idx_set_dict.values()):
        n_keep += 1
        test_idx_set_dict[rand_seed] = set(y_test.index)
    n_try += 1

########################################################################
## bootstrap: run 
########################################################################

import warnings

warnings.filterwarnings('ignore')

log_grid = [{
    'C': [1e-4, 1e-03, 1e-2, 1e-1, 1, 10],
    'l1_ratio': [0.2, 0.4, .6, .8, 1.0],
}]

split_rand_seeds = list(test_idx_set_dict.keys())
features_list = []
test_accuracy_scores = []
model_tune_history = []

print(DATE(), "=== Starting {0} bootstraps ===".format(len(split_rand_seeds)))
for i in range(len(split_rand_seeds)):
    if i % 5 == 0:
        print(DATE(), "running bootstrap #{}".format(i))
    rnd_seed = split_rand_seeds[i]
    test_idx_set = test_idx_set_dict[rnd_seed]
    train_idx_set = set(X.index) - test_idx_set
    X_train = X.loc[train_idx_set]
    y_train = y.loc[train_idx_set, ['stage']]
    X_test = X.loc[test_idx_set]
    y_test = y.loc[test_idx_set, ['stage']]

    log_model = tune_elasticnet(X_train,
                                y_train,
                                rnd_seed,
                                log_grid,
                                n_core=10)
    model_tune_history.append(log_model)

    #print("Best Parameters:\n", log_model.best_params_)

    # Select best log model
    best_log = log_model.best_estimator_

    # Make predictions using the optimised parameters
    log_pred = best_log.predict(X_test)
    test_accuracy_scores.append(
        [log_model.best_score_,
         accuracy_score(y_test, log_pred)])
    #     cm_log = confusion_matrix(y_test, log_pred)

    #features
    idx = np.where(np.ravel(best_log.coef_) != 0)
    features_list.append(
        pd.DataFrame({
            "gene": X.columns[idx],
            "coef": np.ravel(best_log.coef_)[idx]
        }).sort_values('coef', ascending=False))


########################################################################
## Result summarization 
########################################################################

# store weights vector to dict
feature_weight_dict = {
    g: [0 for i in range(100)]
    for g in list(set().union(
        *list(map(lambda x: set(x.gene), features_list))))
}

for i, l in enumerate(features_list):
    for _, row in l.iterrows():
        feature_weight_dict[row['gene']][i] = row['coef']

# add std
df_feature_genes = pd.DataFrame({
    'gene': [g for g in feature_weight_dict.keys()],
    'counts': [np.sum(np.array(v) != 0) for v in feature_weight_dict.values()],
    'avg_weight': [np.sum(np.array(v)) for v in feature_weight_dict.values()],
    'std_weight':
    [np.std(np.array(v)) * 100 for v in feature_weight_dict.values()],
}).set_index('gene').sort_values('avg_weight', ascending=False)

# save results 
import pickle

bootstrap_result_dict = {
    'model_tune_history': model_tune_history,
    'test_accuracy_scores': test_accuracy_scores,
    'features_list': features_list,
    'init_random_seed': init_seed,
}
pickle.dump(bootstrap_result_dict, open('bootstrap_result_dict.p', 'wb'),
            pickle.HIGHEST_PROTOCOL)