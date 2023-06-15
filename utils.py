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