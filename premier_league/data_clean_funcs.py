# This script contains the functions used to clean the premier league data
import pandas as pd

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


def convert_result_to_win(goals_for, goals_against):
    """
    This functions calculates whether a team won based on goals for and goals against
    """
    if goals_for > goals_against:
        return 1
    else:
        return 0