import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.inspection import permutation_importance
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# To-do: Write a wrapper function that runs the feature importance analysis

def select_cols_within_dict(dict, keys, cols_to_select):
    for key in keys:
        df = dict[key][cols_to_select]
        dict[key] = df
    return dict


def pull_col_indexes(names, cols_oi):
    return [i for i in range(len(names)) if names[i] in cols_oi]


def implement_feature_selection(dict, col_indicies, keys):
    """
    This function edits the test/train dictionary to only keep the selected features
    """
    for key in keys:
        dict[key] = np.take(dict[key], col_indicies, axis=1)
    return dict


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
    forest = RandomForestClassifier(n_estimators=100, random_state=0)
    forest.fit(x_train, y_train)
    return forest


def calc_array_cum_sum(array):
    """This function sorts an array largest to smallest and calculates the cumulative sum"""
    if isinstance(array, pd.Series):
        sorted = array.sort_values(ascending=False)
        return pd.Series(sorted.cumsum(), index=list(sorted.index))
    else:
        return np.sort(array)[::-1].cumsum()


def calc_cum_sum_threshold(array, threshold):
    """This function calculates the cumulative sum of an array and filters the original array once it hits a certain proportion of the cumulative sum"""
    cum_sum_array = calc_array_cum_sum(array)
    total = max(cum_sum_array)
    cutoff_index = np.argmax(cum_sum_array >= threshold * total)
    reduced_array = cum_sum_array[:cutoff_index+1]
    return reduced_array


def pull_features_needed_to_hit_importance(features_series, threshold):
    reduced = calc_cum_sum_threshold(features_series, threshold)
    features = list(reduced.index)
    coverage = max(reduced)*100
    print("The following features have been selected:")
    print("\n".join(features))
    print(f"These features explain {coverage:.2f}% of relative importance")
    return features


def plot_cum_importance(series, threshold):
    sorted = series.sort_values(ascending=False)
    summed = calc_array_cum_sum(sorted)
    x_locations = list(range(len(sorted)))
    plt.figure()
    plt.plot(x_locations, summed)
    plt.hlines(y = threshold, xmin=0, xmax=len(series), color = 'r', linestyles = 'dashed')
    plt.xticks(x_locations, list(summed.index), rotation = 'vertical')
    plt.xlabel('Feature'); plt.ylabel('Cumulative Importance'); plt.title('Cumulative Importances')


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


def evaluate_classifier_model(inputs, model_fn, **model_params):
    """
    This function takes takes in a dictionary of test and training datasets and predicts the target variable using a specified model and parameters
    Args:
        inputs: a dictionary with keys x_train, x_test, y_train, y_test as keys and values are associated arrays/dataframes
        model_fn: name of the classifier function to run
        model_params: a list of additional arguments to pass
        """
    print(f"Running a {model_fn.__name__} model ...")
    model = model_fn(**model_params)

    # Fit the model to the training data
    model.fit(inputs['x_train'], inputs['y_train'])

    # Predict the target variable for the test data
    y_pred = model.predict(inputs['x_test'])

    # Calculate the accuracy score
    accuracy = accuracy_score(inputs['y_test'], y_pred)

    return model, y_pred, accuracy
