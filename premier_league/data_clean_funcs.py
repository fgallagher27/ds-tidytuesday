# This script contains the functions used to clean the premier league data
import pandas as pd
import numpy as np

def expand_cols_to_for_and_against(df, col_dicts):
    """
    This function takes a pair of columns with a metric for the home team and a metric for the away team.
    It then creates 2 new columns wiht output columns indicating home_for, home_against, away_for, away_against
    Args:
        df: an input pandas DataFrame
        col_dicts: a list of dictionaries where each key is a variable
            and each set value is a list of 2 column names:
            first name indicates home observation of the variable, the latter indicates the away value
    """
    new_dfs = []
    for col_dict in col_dicts:
        var_name = list(col_dict.keys())[0]
        cols = col_dict[var_name]
        home_col = cols[0]
        away_col = cols[1]

        home_for = away_against = df[home_col]
        home_against = away_for = df[away_col]

        new_df = pd.DataFrame({
            f'home_{var_name}_for': home_for,
            f'home_{var_name}_against': home_against,
            f'away_{var_name}_for': away_for,
            f'away_{var_name}_against': away_against
        })

        new_dfs.append(new_df)

    final_df = pd.concat(new_dfs, axis=1)
    
    return final_df


def pivot_home_away_to_long(df, id_cols):
    """
    This function takes the wide dataframe with home and away columns for each variable and
    melts and pivots to get one column for each variable and a binary column for whether the
    team played that match home or away.

    Args:
        df: a pandas dataframe
        id_cols: the columns that identify each match e.g. date and teams
    """
    home_cols = ['date', 'home_team'] + np.setdiff1d([item for item in df.columns if not item.startswith('away_')], id_cols).tolist()
    away_cols = ['date', 'away_team'] + np.setdiff1d([item for item in df.columns if not item.startswith('home_')], id_cols).tolist()
   
    temp = df.melt(id_vars=id_cols, var_name='statistic')

    # split the statistic column into two columns
    temp[['home_away', 'statistic']] = temp['statistic'].str.split('_', n=1, expand=True)

    # pivot the dataframe
    temp = temp.pivot_table(index=id_cols, columns=['home_away', 'statistic'], values='value').reset_index()
    temp.columns = temp.columns.map('_'.join).str.rstrip('_')

    # rename the columns
    temp = temp.rename(columns={
        'Date': 'date',
        'HomeTeam': 'home_team',
        'AwayTeam': 'away_team'
    })

    # create a new dataframe with the home team data
    df_home = temp[home_cols].copy()
    df_home.columns = df_home.columns.str.replace('home_', '')
    # create 'home' indicator after 'team'
    idx = df_home.columns.get_loc('team') + 1
    df_home.insert(loc=idx, column='home', value=1)

    # same for away data
    df_away = temp[away_cols].copy()
    df_away.columns = df_away.columns.str.replace('away_', '')
    # create indicator for away by setting 'home' = 0
    idx = df_away.columns.get_loc('team') + 1
    df_away.insert(loc=idx, column='home', value=0)

    df_final = pd.concat([df_home, df_away], ignore_index=True)
    return df_final


def convert_result_to_win(goals_for, goals_against):
    """
    This functions calculates whether a team won based on goals for and goals against
    """
    def inner_check(row):
        if row[goals_for] > row[goals_against]:
            return 1
        else:
            return 0
    return inner_check