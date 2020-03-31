import numpy as np
import pandas as pd
import imblearn
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import BernoulliNB
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import classification_report

def split_and_resample(X,y):
    '''Split dataset into train and test, and resample the imbalanced classes'''
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state=47)
    undersample = imblearn.under_sampling.RandomUnderSampler(random_state=15, replacement=False)
    X_train_resampled, y_train_resampled = undersample.fit_resample(X_train, y_train)
    return X_train_resampled, X_test, y_train_resampled, y_test

def fit_shallow_models(X_train, X_test, y_train, y_test):
    '''Fit two candidate models to classify concepts. Print overall model performance and class performance'''
    classes = np.unique(y_train)
    model_names = ["nb", "sgd"]
    nb = BernoulliNB(alpha=0.001, binarize=0.0, fit_prior=True, class_prior=None)
    sgd = SGDClassifier(loss='hinge', penalty='l2', alpha=0.0001, l1_ratio=0.15, fit_intercept=True, 
                  max_iter=1000, tol=0.001, shuffle=True, verbose=1, epsilon=0.1, 
                  n_jobs=None, random_state=None, learning_rate='optimal', eta0=0.0, 
                  power_t=0.5, early_stopping=False, validation_fraction=0.1, n_iter_no_change=5, 
                  class_weight=None, warm_start=False, average=False)
    '''Train and test models, outputting reports on model performance'''
    for model_name in model_names:
        curr_model = eval(model_name)
        curr_model.partial_fit(X_train, y_train, classes)
        print(f'{model_name} model score: {curr_model.score(X_test, y_test)}')
        print(f'{model_name} classification performance:',classification_report(y_pred=curr_model.predict(X_test), y_true=y_test, labels=classes))
    return 
        
    