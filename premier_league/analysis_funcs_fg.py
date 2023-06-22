import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.inspection import permutation_importance
from sklearn.ensemble import RandomForestClassifier

# To-do: Write a wrapper function that runs the feature importance analysis


def isolate_target_and_predictor_vars(df, target_var, predictor_vars):
    """
    This function takes a pandas dataset and splits it into the target variable and the predicting variables.
    Args:
        df: a pandas dataframe
        target_var: a chr string for the target_variable
        predictor_vars: a list of column names referring to the predictor variables
    """
    target = df[target_var]
    predictor = df.loc[:, predictor_vars]
    return target, predictor


def fit_random_classifier(x_train, y_train):
    """This function conducts a random forest classifier and fits the x and y training sets"""
    print("Fitting RandomForestClassifier...")
    forest = RandomForestClassifier(random_state=0)
    forest.fit(x_train, y_train)
    return forest


def calc_importance_mdi(forest):
    """
    This function calculates the feature importance based on Mean Decrease in Impurity.
    Args:
        forest: the output of a fitted Random Forest Classifier
    """
    print("Computing importances based on MDI...")

    start_time = time.time()
    importances = forest.feature_importances_
    std = np.std([tree.feature_importances_ for tree in forest.estimators_], axis=0)
    elapsed_time = time.time() - start_time

    print(f"Time taken to compute the importances based on MDI: {elapsed_time:.3f} seconds")

    return importances, std


def calc_importance_permuation(forest, x_test, y_test):
    """
    This function calculates the feature importance based on permuation across the full model.
    Args:
        forest: the output of a fitted Random Forest Classifier
        x_test: the testing set for the predicting variables
        y_test: the testing set for the target variable
    
    """
    # To-do: edit function arguments to take additional arguments to permutation_importance as if ... in R
    print("Computing importances based on permutation...")
    start_time = time.time()
    result = permutation_importance(
        forest, x_test, y_test, n_repeats=10, random_state=42, n_jobs=2
    )
    elapsed_time = time.time() - start_time

    print(f"Elapsed time to compute the importances based on permutation: {elapsed_time:.3f} seconds")

    return result.importances_mean, result.importances_std


def plot_feature_importance(importances, std, feature_names, labels):
    """
    This function plots the relative importance of features as a bar chart with standard deviation errors.
    Args:
        importances: the output of a random forest feature importance run
        std: the standard error of the output
        labels: a dictionary containing a plot 'title' and a plot 'ylabel'
    """
    importance_series = pd.Series(importances, index=feature_names)
    fig, ax = plt.subplots()
    importance_series.plot.bar(yerr=std, ax=ax)
    ax.set_title(labels['title'])
    ax.set_ylabel(labels['ylabel'])
    fig.tight_layout()
