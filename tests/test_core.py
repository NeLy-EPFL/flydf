import pandas as pd
import pytest
import numpy as np

import flydf

@pytest.fixture
def test_df():
    df = pd.DataFrame(columns=["Genotype", "Date", "Fly", "Trial", "Frame"])
    for genotype in ["R57C10", "SS123245"]:
        for date in [191220, 200313]:
            for fly in [1, 3]:
                for trial in [4, 5]:
                    data = np.random.rand(10, 4)
                    df = flydf.add_data(df, genotype, date, fly, trial, [f"Column {i}" for i in range(data.shape[1])], data)
    return df


def check_df_values(df, data, genotype, date, fly, trial, columns):
    for frame in range(data.shape[0]):
        df_frame = df.loc[(df["Genotype"] == genotype) &
                          (df["Date"] == date) & 
                          (df["Fly"] == fly) & 
                          (df["Trial"] == trial) &
                          (df["Frame"] == frame)
                         ]
        for i, col in enumerate(columns):
            index = df_frame.index.values.astype(int)[0]
            df_value = df_frame.at[index, col]
            assert df_value == data[frame, i]


def test_add_data(test_df):
    df = pd.DataFrame(columns=["Genotype", "Date", "Fly", "Trial", "Frame"])
    for genotype in ["R57C10", "SS123245"]:
        for date in [191220, 200313]:
            for fly in [1, 3]:
                for trial in [4, 5]:
                    data = np.random.rand(3, 4)
                    df = flydf.add_data(df, genotype, date, fly, trial, [f"Column {i}" for i in range(data.shape[1])], data)
    assert df.shape[0] == 16 * data.shape[0]
    assert df.shape[1] == data.shape[1] + len(flydf.default_columns)
    assert not df.duplicated(subset=flydf.default_columns).any()

    data = np.random.rand(5, 6)
    genotype = "PR"
    date = 200514
    fly = 1
    trial = 1
    columns = [f"Column {i}" for i in range(data.shape[1])]
    df = flydf.add_data(df, genotype, date, fly, trial, columns, data)
    check_df_values(df, data, genotype, date, fly, trial, columns)

    # Check that additional columns added are filled with NaNs for the data that was already present
    previous_df = df.loc[(df["Genotype"] != genotype) &
                         (df["Date"] != date) & 
                         (df["Fly"] != fly) & 
                         (df["Trial"] != trial),
                         [f"Column {i}" for i in range(4, 6)]
                        ]
    assert previous_df.isnull().values.all()

    # Replace NaN values of columns that were automatically filled.
    genotype = "R57C10"
    date = 191220
    fly = 1
    trial = 4
    nan_columns = [f"Column {i}" for i in range(4, 6)]
    nan_data = np.random.rand(3, 2)
    old_columns = [f"Column {i}" for i in range(4)]
    old_data = flydf.get_trial_df(df, genotype, date, fly, trial)[old_columns].values
    df = flydf.add_data(df, genotype, date, fly, trial, nan_columns, nan_data)
    data = np.concatenate([old_data, nan_data], axis=1)
    columns = old_columns + nan_columns
    check_df_values(df, data, genotype, date, fly, trial, columns)

    # Raises error when you try to add duplicates with differing values
    with pytest.raises(ValueError):
        df = flydf.add_data(df, genotype, date, fly, trial, nan_columns, nan_data + 1)
    partial_overlap_columns = ["Column 4", "Column 6"]
    with pytest.raises(ValueError):
        df = flydf.add_data(df, genotype, date, fly, trial, partial_overlap_columns, nan_data + 1)
    # Doesn't raise when duplicate values are the same
    df = flydf.add_data(df, genotype, date, fly, trial, partial_overlap_columns, nan_data)

    # Add additional columns to existing data
    data = np.random.rand(3, 2)
    old_number_of_entries = df.shape[0]
    columns = [f"New column {i}" for i in range(data.shape[1])]
    df = flydf.add_data(df, genotype, date, fly, trial, columns, data)
    new_number_of_entries = df.shape[0]
    assert old_number_of_entries == new_number_of_entries
    check_df_values(df, data, genotype, date, fly, trial, columns)


def test_split_into_genotype_dfs(test_df):
    for genotype_df in flydf.split_into_genotype_dfs(test_df):
        genotypes = list(set(genotype_df["Genotype"].values))
        assert len(genotypes) == 1
        genotype = genotypes[0]
        genotype_df.equals(test_df[test_df["Genotype"] == genotype])


def test_split_into_fly_dfs(test_df):
    for fly_df in flydf.split_into_fly_dfs(test_df):
       
        genotypes = list(set(fly_df["Genotype"].values))
        assert len(genotypes) == 1
        genotype = genotypes[0]
        
        dates = list(set(fly_df["Genotype"].values))
        assert len(dates) == 1
        date = dates[0]
        
        flys = list(set(fly_df["Genotype"].values))
        assert len(flys) == 1
        fly = flys[0]
        
        fly_df.equals(test_df[(test_df["Genotype"] == genotype) &
                              (test_df["Date"] == date) &
                              (test_df["Fly"] == fly)
                             ]
                     )

        
def test_split_into_trial_dfs(test_df):
    for trial_df in flydf.split_into_trial_dfs(test_df):
       
        genotypes = list(set(trial_df["Genotype"].values))
        assert len(genotypes) == 1
        genotype = genotypes[0]
        
        dates = list(set(trial_df["Genotype"].values))
        assert len(dates) == 1
        date = dates[0]
        
        flys = list(set(trial_df["Genotype"].values))
        assert len(flys) == 1
        fly = flys[0]

        trials = list(set(trial_df["Trial"].values))
        assert len(trials) == 1
        trial = trials[0]
        
        trial_df.equals(test_df[(test_df["Genotype"] == genotype) &
                              (test_df["Date"] == date) &
                              (test_df["Fly"] == fly) &
                              (test_df["Trial"] == trial)
                             ]
                     )

def test_split_into_epoch_dfs(test_df):
    for epoch_df in flydf.split_into_epoch_dfs(test_df):
       
        genotypes = list(set(epoch_df["Genotype"].values))
        assert len(genotypes) == 1
        genotype = genotypes[0]
        
        dates = list(set(epoch_df["Genotype"].values))
        assert len(dates) == 1
        date = dates[0]
        
        flys = list(set(epoch_df["Genotype"].values))
        assert len(flys) == 1
        fly = flys[0]

        trials = list(set(epoch_df["Trial"].values))
        assert len(trials) == 1
        trial = trials[0]
       
        epoch_df = epoch_df.sort_values("Frame", axis="index")

        frame_to_frame_intervals = np.diff(epoch_df["Frame"].values)
        assert len(set(frame_to_frame_intervals)) == 1
        assert np.isclose(frame_to_frame_intervals[0], 1)

        start_frame = epoch_df.at[epoch_df.index[0], "Frame"]
        stop_frame = epoch_df.at[epoch_df.index[-1], "Frame"]

        # Check that previous and next frames don't exist
        trial_bool_index = ((test_df["Genotype"] == genotype) &
                           (test_df["Date"] == date) &
                           (test_df["Fly"] == fly) &
                           (test_df["Trial"] == trial))
        assert not np.any(trial_bool_index & (test_df["Frame"] == start_frame - 1))
        assert not np.any(trial_bool_index & (test_df["Frame"] == stop_frame + 1))

        epoch_df.equals(test_df[(test_df["Genotype"] == genotype) &
                              (test_df["Date"] == date) &
                              (test_df["Fly"] == fly) &
                              (test_df["Trial"] == trial) &
                              (test_df["Frame"] >= start_frame) &
                              (test_df["Frame"] <= stop_frame)
                             ]
                     )


def test_number_of_epochs(test_df):
    n = flydf.number_of_epochs(test_df)
    n_trials = 0
    for trial_df in flydf.split_into_trial_dfs(test_df):
        n_trials += 1
    assert n == n_trials

    # Split all trials into two epochs
    test_df = test_df[test_df["Frame"] != 5]
    n = flydf.number_of_epochs(test_df)
    assert n == 2 * n_trials
