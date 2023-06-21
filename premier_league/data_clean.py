import pandas as pd
import pickle
from data_clean_funcs import *

premier_leauge = pd.read_csv("https://raw.githubusercontent.com/rfordatascience/tidytuesday/master/data/2023/2023-04-04/soccer21-22.csv")

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
    'HY': 'home_yellows',
    'AY': 'away_yellows',
    'HR': 'home_reds',
    'AR': 'away_reds'
})

full_wide_df = pd.concat([subset_pl_df, expanded_df], axis=1)

ids = ['Date', 'HomeTeam', 'AwayTeam']

long_df = pivot_home_away_to_long(full_wide_df, ids)

long_df['ft_result'] = long_df.apply(lambda row: convert_result_to_win('ft_goals_for', 'ft_goals_against')(row), axis=1)
long_df['ht_result'] = long_df.apply(lambda row: convert_result_to_win('ht_goals_for', 'ht_goals_against')(row), axis=1)

pickle_out = open("data.pickle","wb")
pickle.dump(long_df, pickle_out)
pickle_out.close()
