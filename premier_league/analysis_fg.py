import pickle
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import scale
from analysis_funcs_fg import *

# Load in pickled dataframe
pickle_in = open("long_premier_leauge_df.pickle","rb")
prem_df = pickle.load(pickle_in)

# assign parameter values - TODO move parameters to a config file
target = 'ft_result'
non_predictors = ['date', 'team', 'ft_goals_for', 'ft_goals_against', 'ht_goals_for', 'ht_goals_against', target]
predictors = [item for item in list(prem_df) if item not in non_predictors]
relative_importance_threshold = 0.9
show_plots = False

# split predictors and target variables
prem_results, prem_stats = isolate_target_and_predictor_vars(prem_df, target, predictors)
feature_names = list(prem_stats)

# standardise predictors
prem_stats_scaled = scale(prem_stats, axis=0)

# create test and training splits
x_train, x_test, y_train, y_test = train_test_split(prem_stats_scaled, prem_results, random_state=42)

data_dict = {
        'x_train': x_train,
        'x_test': x_test,
        'y_train': y_train,
        'y_test': y_test
    }

# logistic baseline
log_model, log_predict, log_accuracy = evaluate_classifier_model(data_dict, LogisticRegression)

# run and fit random forest classifier
forest_classifier = fit_random_classifier(x_train, y_train)

# calc feature importance scores by mean decrease in impurity
# this method is biased towards features with more categories
# it also has no preference over which of two (or more) correlated features should be chosen
# i.e. half_time_goals_for and half_time_result
# this is not an issue for reducing overfitting, but need to be careful interpreting the resulting relative importances
mdi_importance, mdi_std = calc_importance_mdi(forest_classifier)

# Feature permutation is not biased towards high cardinality features, but is more likely to miss out a feature entirely
perm_importance, perm_std = calc_importance_permuation(
    forest_classifier,
    x_test,
    y_test)

# plot feature importance scores by method
plot_feature_importance(
    mdi_importance,
    mdi_std,
    feature_names,
    labels={
        'title': 'Feature importances using MDI',
        'ylabel': 'Mean Decrease in Impurity (MDI)'
    })

plot_feature_importance(
    perm_importance,
    perm_std,
    feature_names,
    labels={
        'title': 'Feature importances using permuation on full model',
        'ylabel': 'Mean accuracy decrease'
    })


feature_series = pd.Series(mdi_importance, index=feature_names)
plot_cum_importance(feature_series, relative_importance_threshold)

# TODO either show plots and ask user to choose mdi vs perm, or convert to notebook so user can run sections and see the plots
plt.show(block=show_plots)

# TODO: create function that benchmarks accuracy and run time depending on number of features

# train and evaluate Random Forest Classifier
rf_model, rf_predict, rf_accuracy = evaluate_classifier_model(
        data_dict,
        RandomForestClassifier,
        n_estimators=200,
        random_state=42)

print(f"Logistic Regression has accuracy of: {log_accuracy:.3f}")
print(f"Random Forest Classifier has accuracy of: {rf_accuracy:.3f}")

# now implement feature selection
selected_features = pull_features_needed_to_hit_importance(feature_series, relative_importance_threshold)
# TODO options to pull features based on a mean decrease in accuracy criteria instead

selected_feature_index = pull_col_indexes(list(prem_stats), selected_features)
selected_data_dict = implement_feature_selection(data_dict, selected_feature_index, ['x_train', 'x_test'])

selected_rf_model, selected_rf_predict, selected_rf_accuracy = evaluate_classifier_model(
        selected_data_dict,
        RandomForestClassifier,
        n_estimators=200,
        random_state=42)

print(f"Random Forest Classifier with feature selection has accuracy of: {selected_rf_accuracy:.3f}")
