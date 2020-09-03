import pandas as pd
import numpy as np

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
        raise ValueError(f"df need to have columns {list(default_columns)}")

    existing_columns = set(column_names).intersection(set(df.columns))
    new_columns = set(column_names).difference(set(df.columns))

    if not existing_columns.issubset(set(default_columns)):
        df = df.append(new_data_df[existing_columns.union(set(default_columns))], ignore_index=True, sort=False)
        if df.duplicated(subset=list(default_columns)).any():
            raise ValueError("Contains duplicates")
    
    if len(new_columns) > 0:
        df = pd.merge(df, new_data_df[new_columns.union(set(default_columns))], on=list(default_columns), how="outer", validate="one_to_one")

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
