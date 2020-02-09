"""
These are other miscellaneous functions necessary to build the neural forest model. 
Author:
    Sabareesh Mamidipaka
Date:
    12/25/2018
"""

import pandas as pd

class standardscaler(object):
    """
    Scale the data
    """
    def __init__(self, data):
        self.mean = np.mean(data)
        self.scale = np.std(data)
        
    def transform(self, data):
        return (data-self.mean)/self.scale
    
    def inversetransform(self, data):
        return data*self.scale+self.mean


def bag_data(data: pd.DataFrame, col_index_list: list, target_column: list):
    """
    Selects the subset of the data according to the index-list.
    performs bagging and returns training data, training label, oob data and oob labels as arrays.
    """
    
    # Slice the subset of data we want to work with (features + target) 
    data = data.iloc[:,list(col_index_list)+list(data.columns.get_loc(c) for c in target_column)]
    
    # Remove the rows which contain null values
    data = data.dropna()

    # Sample the indices to perform bagging
    n_rows = len(data)
    train_index = np.random.randint(n_rows, size=n_rows)
    oob_index = np.setdiff1d(np.arange(n_rows),train_index)
    
    # Split the data into training and validation (oob samples)
    df_train = data.iloc[train_index]
    df_val = data.iloc[oob_index]
    
    print(df_train.columns)

    # Convert the training data into training and target arrays
    train_x = df_train.drop(target_column,axis=1).values
    train_y = df_train[target_column].values

    # Convert the validation into training and label arrays
    val_x = df_val.drop(target_column,axis=1).values
    val_y = df_val[target_column].values

    return train_x, train_y, val_x, val_y


def make_features_list(columns, num_features, target_col):
    
    n_columns = data.columns.nunique()
    iterable = set(np.arange(n_columns))
    target_col_index = set(columns.get_loc(c) for c in target_col)
    features_index = iterable-target_col_index
    return list(itertools.combinations(features_index, num_features))
    
