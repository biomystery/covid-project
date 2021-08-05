#!/usr/bin/env python3
# import sys, os
import pandas as pd
import numpy as np
import datetime
import argparse
DATE = lambda: datetime.datetime.now().strftime("[%m/%d/%Y %H:%M:%S]")

########################################################################
## read input 
########################################################################
def get_args():
    """
    process the input arguments 
    """
    parser = argparse.ArgumentParser(description='Plot bootstrap results.')
    parser.add_argument('-i','--input-results', help='result from the bootstrapping',required=True)

    args=parser.parse_args()
    return args 


def load_data(args):
    """
    prepare input data for training 
    """
    # Import labels (for the whole dataset, both training and testing)
    y = pd.read_csv(args.input_y_label, index_col=0)
    print(y.shape)
    y['label'].value_counts()

    X = pd.read_csv(args.input_x_matrix, index_col=0).T
    print(X.shape)    
    if not all(X.index == y.index): raise("inputError: X and y don't match!")
    return X,y


########################################################################
## bootstrap: preparing 
########################################################################
def prepare_bootstrap(args,X,y,init_seed = 200):
    """
    prepare bootstrap.
    Input: 
        - number of bootstraps from args 
        - X 
        - y 
        - init_seed: initial random seed to start bootstrapping 
    Output:
        - test_idx_set_dict: dictionary with key - rand_seed, val - sample index for test set 

    """
    
    # split train and testing
    from sklearn.model_selection import train_test_split
    import random

    n_keep = 1
    n_try = 1
    test_idx_set_dict = {}
    
    random.seed(a=init_seed, version=2)
    while n_keep <= args.number_of_bootstrapping:
        rand_seed = random.randrange(args.number_of_bootstrapping*10, )
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
    return test_idx_set_dict 

########################################################################
## bootstrap: run 
########################################################################

def run_bootstrap(args,X,y,test_idx_set_dict):
    """
    run bootstrapping 
    output: 
        - bootstrap_result_dict: 
            - 'model_tune_history': model_tune_history,
            - 'test_accuracy_scores': test_accuracy_scores,
            - 'features_list': features_list,
            - 'init_random_seed': init_seed
    """
    import warnings
    warnings.filterwarnings('ignore')

    
    from sklearn.metrics import accuracy_score #confusion_matrix

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
        y_train = y.loc[train_idx_set, ['label']]
        X_test = X.loc[test_idx_set]
        y_test = y.loc[test_idx_set, ['label']]

        log_model = tune_elasticnet(X_train,
                                    y_train,
                                    rnd_seed,
                                    log_grid,
                                    n_core=10)
        model_tune_history.append(log_model)

        #print("Best Parameters:\n", log_model.best_params_)

        # Select best log model
        best_log = log_model.best_estimator_

        # Make predictions using the optimized parameters
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

    ## Result summarization 
    print(DATE(), "=== Result summary ===")

    # store weights vector to dict
    feature_weight_dict = {
        g: [0 for i in range(100)]
        for g in list(set().union(
            *list(map(lambda x: set(x.gene), features_list))))
    }

    for i, l in enumerate(features_list):
        for _, row in l.iterrows():
            feature_weight_dict[row['gene']][i] = row['coef']

    # # add std
    # df_feature_genes = pd.DataFrame({
    #     'gene': [g for g in feature_weight_dict.keys()],
    #     'counts': [np.sum(np.array(v) != 0) for v in feature_weight_dict.values()],
    #     'avg_weight': [np.sum(np.array(v)) for v in feature_weight_dict.values()],
    #     'std_weight':
    #     [np.std(np.array(v)) * 100 for v in feature_weight_dict.values()],
    # }).set_index('gene').sort_values('avg_weight', ascending=False)

    # save results 
    if args.save_result_object:
        import pickle
        bootstrap_result_dict = {
            'model_tune_history': model_tune_history,
            'test_accuracy_scores': test_accuracy_scores,
            'features_list': features_list,
            'init_random_seed': init_seed,
        }
        pickle.dump(bootstrap_result_dict, open('bootstrap_result_dict.p', 'wb'),
                    pickle.HIGHEST_PROTOCOL)

    return bootstrap_result_dict

########################################################################
## Main
########################################################################
def main():
    """
    main function
    """
    args = get_args()
    X,y = read_data(args)
    test_idx_set_dict = prepare_bootstrap(args,X,y)

    _ = run_bootstrap(args,X,y,test_idx_set_dict)

if __name__ == "__main__":

    main()