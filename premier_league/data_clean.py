import pandas as pd
import numpy as np
from data_clean_funcs import *

premier_leauge = pd.read_csv("https://raw.githubusercontent.com/rfordatascience/tidytuesday/master/data/2023/2023-04-04/soccer21-22.csv")

home_cols = ['FTHG', 'HTHG', 'HS', 'HST', 'HF', 'HC', 'HY', 'HR']
away_cols = ['FTAG', 'HTAG', 'AS', 'AST', 'AF', 'AC', 'AY', 'AR']
cols = ['ft_goals_for', 'ht_goals_for', 'shots', 'shots_on_target', 'fouls', 'corners', 'yellow_cars', 'red_cards']

ideal_cols = [
    'home',
    'win',
    'ft_goals_for',
    'ft_goals_against',
    'ht_goals_for',
    'ht_goals_against',
    'shots_taken',
    'shots_faced',
    'shots_on_target_taken',
    'shots_on_target_faced',
    'fouls',
    'corners_taken',
    'corners_faced',
    'yellow_cards',
    'red_cards'
    ]

expanded_df = expand_cols_to_for_and_against(
    premier_leauge,
    [
        {'ft_goals': ['FTHG', 'FTAG']},
        {'ht_goals': ['HTHG', 'HTAG']},
        {'shots': ['HS', 'AS']},
        {'shots_on_target': ['HST', 'AST']},
        {'corners': ['HC', 'AC']}
    ])

subset_pl_df = premier_leauge.loc[:, ['Date', 'HomeTeam', 'AwayTeam', 'HF', 'AF', 'HY', 'HR']].rename(columns={
    'HF': 'home_fouls',
    'AF': 'away_fouls',
    'HY': 'home_yellow',
    'AY': 'away_yellow',
    'HR': 'home_red',
    'AR': 'away_red'
})

full_wide_df = pd.concat([subset_pl_df, expanded_df], axis=1)
print(full_wide_df.head(10))


# FTR	character	Full time result

# HTR	character	Halftime results
# Referee	character	Referee of the match


# create team col

# categorisiation problem - label as Win, Not Win

# create binary column for home away

# may need log transformation

# either linear probability or logistic regression

# any kind of classificaton
# random forest - classification alogrith
# Naive bayes classification
# Neural network 

# feature importance procedure
# ridge regression

