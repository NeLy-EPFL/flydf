import pandas as pd
import numpy as np

from tqdm import tqdm


default_columns_dtypes = {
        "Genotype": "category",
        "Date": "category",
        "Fly": "category",
        "Trial": "category",
        "Frame": "unsigned",
        }

default_columns = list(default_columns_dtypes)


def _fix_dtypes(df):
    for col_name, dtype in default_columns_dtypes.items():
        if dtype in ["integer", "signed", "unsigned", "float"]:
            df[col_name] = pd.to_numeric(df[col_name], downcast=dtype)
        else:
            df[col_name] = df[col_name].astype(dtype)
    return df


def _agg_function(series):
    series = series.dropna()
    series = series.drop_duplicates(keep="first")
    if len(series) == 1:
        return series.iloc[0]
    elif len(series) == 0:
        return np.nan
    else:
        raise ValueError("Trying to overwrite existing values with different values.")
    
        

def add_data(df, genotype, date, fly, trial, column_names, data):
    """
    Parameters
    ----------
    df : pandas.DataFrame
        The data frame to which the data should be added.
    genotype : str
        Genotype of the given data.
    date : int
        Date of the given data.
    fly : int
        Fly number.
    trial : int
        Trial number.
    column_names : list of str
        Names of the columns to be added.
    data : numpy array
        Data that should be added.
    
    Returns
    -------
    df : pandas.DataFrame
        Data frame with added data.
    """
    new_data_df = pd.DataFrame(data=data, index=None, columns=column_names)
    new_data_df["Genotype"] = genotype
    new_data_df["Date"] = date
    new_data_df["Fly"] = fly
    new_data_df["Trial"] = trial
    new_data_df["Frame"] = np.arange(len(data))
    
    if not set(default_columns).issubset(set(df.columns)):
        raise ValueError(f"df need to have columns {default_columns}")

    existing_columns = set(column_names).intersection(set(df.columns))
    new_columns = set(column_names).difference(set(df.columns))

    if not existing_columns.issubset(set(default_columns)):
        df = df.append(new_data_df[existing_columns.union(set(default_columns))], ignore_index=True, sort=False)
        if df.duplicated(subset=default_columns).any():
            agg_functions = {}
            #for col in set(df.columns).difference(existing_columns):
            #    agg_functions[col] = "first"
            #for col in set(df.columns).difference(existing_columns):
            #for col in existing_columns.difference(set(default_columns)):
            for col in set(df.columns).difference(set(default_columns)):
                agg_functions[col] = _agg_function
            df = df.groupby(default_columns).agg(agg_functions)
            df = df.reset_index()
    
    if len(new_columns) > 0:
        df = pd.merge(df, new_data_df[new_columns.union(set(default_columns))], on=default_columns, how="outer", validate="one_to_one")

    return _fix_dtypes(df)


def split_into_genotype_dfs(df):
    """
    Returns a generator that yields padas.DataFrames that only
    contain the data of one genotype.
    """
    for genotype in set(df["Genotype"]):
        genotype_df = df[df["Genotype"] == genotype]
        yield genotype_df


def split_into_fly_dfs(df):
    """
    Returns a generator that yields padas.DataFrames that only
    contain the data of one fly.
    """
    for genotype_df in split_into_genotype_dfs(df):
        for date in set(genotype_df["Date"]):
            date_df = genotype_df[genotype_df["Date"] == date]
            for fly in set(date_df["Fly"]):
                fly_df = date_df[date_df["Fly"] == fly]
                yield fly_df


def split_into_trial_dfs(df):
    """
    Returns a generator that yields padas.DataFrames that only
    contain the data of one trial.
    """
    for fly_df in split_into_fly_dfs(df):
        for trial in set(fly_df["Trial"]):
            trial_df = fly_df[fly_df["Trial"] == trial]
            yield trial_df


def split_into_epoch_dfs(df):
    """
    Returns a generator that yields padas.DataFrames that only
    contain the data of one epoch.
    """
    for trial_df in split_into_trial_dfs(df):
        trial_df = trial_df.sort_values("Frame", axis="index")
        trial_df["diff"] = trial_df["Frame"].diff()
        epoch_boundaries = list(np.where(trial_df["diff"] > 1)[0])
        epoch_boundaries = (
            [0,] + epoch_boundaries + [len(trial_df.index),]
        )
        for start, stop in zip(epoch_boundaries[:-1], epoch_boundaries[1:]):
            yield trial_df[start:stop]


def split_into_dfs_by_column(df, column):
    """
    Splits finding different values in a given column.
    """
    if column is None:
        yield df
        return
    levels = df[column].unique()
    for level in levels:
        yield df[df[column] == level]
    return


def add_epoch_column(df, extra_column=None):
    """
    Adds a column with an index for different epochs.
    Extra column can be used to split finer than frames,
    e.g. using annotations.
    """
    index_df = pd.DataFrame()
    index = 0
    for epoch_df in split_into_epoch_dfs(df):
        for extra_column_epoch_df in split_into_dfs_by_column(epoch_df, extra_column):
            extra_column_epoch_df = extra_column_epoch_df[default_columns]
            extra_column_epoch_df["Epoch index"] = index
            index_df = index_df.append(extra_column_epoch_df)
            index += 1
    df = df.merge(index_df, how="outer", on=default_columns)
    return df


def number_of_epochs(df):
    """
    Returns the number of epochs in a pandas.DataFrame.
    """
    n = 0
    for _ in split_into_epoch_dfs(df):
        n += 1
    return n


def n_frame_epochs_only(df, n):
    """
    Returns a pandas.DataFrame with no epochs shorter
    than n frames.
    """
    new_df = pd.DataFrame()
    n_epochs = number_of_epochs(df)
    for epoch_df in tqdm(split_into_epoch_dfs(df), total=n_epochs):
        if epoch_df.shape[0] >= n:
            new_df = new_df.append(epoch_df)
    return new_df


def get_trial_df(df, genotype, date, fly, trial):
    return df[(df["Genotype"] == genotype) &
              (df["Date"] == date) &
              (df["Fly"] == fly) &
              (df["Trial"] == trial)
             ]


def get_trial_information(df):
    if number_of_epochs(df) != 1:
        raise ValueError("DataFrame has more than one epoch.")
    date = df["Date"].iloc[0]
    genotype = df["Genotype"].iloc[0]
    fly = df["Fly"].iloc[0]
    trial = df["Trial"].iloc[0]
    return date, genotype, fly, trial


def get_trial_masks(df):
    for genotype in df["Genotype"].unique():
        genotype_mask = (df["Genotype"] == genotype)
        for date in df.loc[genotype_mask, "Date"].unique():
            date_mask = (df["Date"] == date) & genotype_mask
            for fly in df.loc[date_mask, "Fly"].unique():
                fly_mask = (df["Fly"] == fly) & date_mask
                for trial in df.loc[fly_mask, "Trial"].unique():
                    trial_mask = (df["Trial"] == trial) & fly_mask
                    yield trial_mask
