import pickle
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import scale
from analysis_funcs_fg import *

# Load in pickled dataframe
pickle_in = open("long_premier_leauge_df.pickle","rb")
prem_df = pickle.load(pickle_in)

# assign parameter values
target = 'ft_result'
non_predictors = ['date', 'team', 'ft_goals_for', 'ft_goals_against', target]
predictors = [item for item in list(prem_df) if item not in non_predictors]

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

# calc feature importance scores by method
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
plt.show(block=False)

# train and evaluate Random Forest Classifier
rf_model, rf_predict, rf_accuracy = evaluate_classifier_model(
        data_dict,
        RandomForestClassifier,
        n_estimators=200,
        random_state=42)

print(f"Logistic Regression has accuracy of: {log_accuracy}")
print(f"Random Forest Classifier has accuracy of: {rf_accuracy}")

# now implement feature selection
selected_features = ['shots_on_target_against', 'shots_on_target_for', 'ht_result', 'ht_goals_for', 'reds']
selected_feature_index = pull_col_indexes(list(prem_stats), selected_features)
selected_data_dict = implement_feature_selection(data_dict, selected_feature_index, ['x_train', 'x_test'])

selected_rf_model, selected_rf_predict, selected_rf_accuracy = evaluate_classifier_model(
        selected_data_dict,
        RandomForestClassifier,
        n_estimators=200,
        random_state=42)

print(f"Random Forest Classifier with feature selection has accuracy of: {selected_rf_accuracy}")
