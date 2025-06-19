import pandas as pd

def clean_data(
    dataset: pd.DataFrame, 
    target_column: str, 
    date_column: str, 
    dims: tuple[str], 
    countries_list: list[str]
) -> pd.DataFrame:
    """
    Clean and preprocess the input dataset for media mix modeling.

    This function filters out specific impression types, retains only relevant columns,
    aggregates data by date and specified dimensions, and normalizes column names to lowercase.
    
    Parameters
    ----------
    dataset : pd.DataFrame
        The raw input dataset containing media impression data.
    target_column : str
        Column name of the target variable (e.g., conversions, revenue).
    date_column : str
        Column name containing date information.
    dims : tuple[str]
        Dimension columns to group by (e.g., market, channel).
    countries_list : list[str]
        List of countries/markets to filter for in the final dataset.
    
    Returns
    -------
    pd.DataFrame
        Processed dataset with filtered impression columns, grouped by date and 
        dimensions, with all values aggregated and filtered to specified countries.
    """
    # Lower case all column names
    dataset.columns = dataset.columns.str.lower()
    
    # Filter impression columns based on lowercase patterns
    impression_columns = [col for col in dataset.columns if "_imp" in col and 
                      "android" not in col and "web" not in col and 
                      "other" not in col and "ios" not in col and
                      "_app_" not in col]
    
    # Further filter out specific impression types
    excluded_patterns = ["paidm_dsp_imp", "paidm_search_generic_imp", "paidm_search_brand_imp", 
                         "visa", "paidm_soc_twitter_tiktok_imp", "paidm_soc_facebook_imp", 
                         "paidm_soc_instagram_imp", "paidm_soc_linkedin_imp", "paidm_soc_youtube_imp"]
    
    impression_columns = [col for col in impression_columns 
                          if not any(pattern in col for pattern in excluded_patterns)]

    # Create a copy of the data with only the relevant columns
    data_subset = dataset[[date_column, target_column] + list(dims) + impression_columns].copy()
    # Group by date and market and sum the impression columns and target variable
    data_subset = data_subset.groupby([date_column, *dims]).sum().reset_index()

    return data_subset[data_subset["market"].isin(countries_list)]