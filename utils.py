# util functions for data cleaning/manipulation

def check_na(cols, input_df):
    """
    check columns of a dataframe to ensure no NA values
    Args:
        cols: list of column names
        input_df: pandas dataframe
    """
    for col in cols:
        if len(input_df[input_df[col].isna()]) > 0:
            raise ValueError(f"{col} contains NA values")
        
def count_nas(cols, input_df):
    """
    Counts the number of NA observations in a dataframe column
    """
    for col in colS:
        Warning(f"{col} has {len(input_df[input_df[col].isna()])} NA observations")