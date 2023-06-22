# util functions for data cleaning/manipulation

def check_na(cols, input_df):
    """
    check columns of a dataframe to ensure no NA values
    Args:
        cols: list of column names
        input_df: pandas dataframe
    """
    cols_w_NAs = []
    for col in cols:
        if len(input_df[input_df[col].isna()]) > 0:
            cols_w_NAs.append(col)
    raise ValueError("The following columns have NAs:\n" + '\n'.join([col for col in cols_w_NAs]))
        
def count_nas(cols, input_df):
    """
    Counts the number of NA observations in a dataframe column
    """
    for col in cols:
        Warning(f"{col} has {len(input_df[input_df[col].isna()])} NA observations")
