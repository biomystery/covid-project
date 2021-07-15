## Method 

We applied an elastic-net Logistic regression combined with bootstrapping to derive predictive models and robust coefficients of the genes in making predictions {Zou, H. and Hastie, T. J. R. Stat. Soc. B 2005, Barretina et al. Nature 2012}. 

The bootstrapping was done by different data splitting. On each bootstrap, the data was split into training and testing sets with a ratio 4:1 using the train_test_split function from sklearn with "sratify=y" and a random seed generated using randrange function from random package. We generated 100 different splits by repeatedly calling randrange function in train_test_split and duplicated splits are programmingly removed.

For each split, we performed a combination of 3-fold cross-validation and hyperparameters search by using GridSearchCV function with "cv=3" from sklearn on training set. The two hyperparameters, "C" and "l1_ratio", were the targets of the grid search. The "C" parameter is the inverse of lambda, which controls the overall strength of the regulation term in elastic net, and was set to 1e-4, 1e-3, 1e-2, 1e-1, 1 and 10. The "l1_ratio" controls the ratio between L1 and L2 regulation terms, which was 0.2, 0.4, 0.6, 0.8 and 1. The combination of these two hyperparameters with the lowest crossvalidation error was selected and used to derive the coefficient for each gene in the model to predict the early or late stage for each sample in validation set and test set. 

The validation accuracy was reported as training accuracy and the test accuracy was calculated by using accuracy_score function from sklearn (FigxxA) by using the best model on testing set. The coefficients of the genes from these 100 bootstraps were summarized and only genes with at least 90 of 100 bootstraps with a nonzero weight were reported in the final list (n=21). The average coefficient and standard deviation of the coefficients were reported in FigxxB.

