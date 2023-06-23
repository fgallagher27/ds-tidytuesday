import pickle
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from analysis_funcs_fg import *

# Load in pickled dataframe
pickle_in = open("long_premier_leauge_df.pickle","rb")
prem_df = pickle.load(pickle_in)

# assign parameter values
target = 'ft_result'
non_predictors = ['date', 'team', 'ft_goals_for', 'ft_goals_against', target]

# split df into target variable and predictors and create test and train sets
predictors = [item for item in list(prem_df) if item not in non_predictors]
prem_results, prem_stats = isolate_target_and_predictor_vars(prem_df, target, predictors)
feature_names = list(prem_stats)
x_train, x_test, y_train, y_test = train_test_split(prem_stats, prem_results, random_state=42)

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