import numpy as np
import pandas as pd

def rolling_slices(df: pd.DataFrame, time_col: str, train_end: str, val_end: str, step: str = "30D"):
    # yields (train_idx, val_idx, test_idx) for walk-forward evaluation
    dates = df[time_col].sort_values()
    start = dates.min()
    cur_train_end = pd.to_datetime(train_end)
    cur_val_end = pd.to_datetime(val_end)
    while cur_val_end < dates.max():
        train_mask = (df[time_col] <= cur_train_end)
        val_mask = (df[time_col] > cur_train_end) & (df[time_col] <= cur_val_end)
        test_end = cur_val_end + pd.Timedelta(step)
        test_mask = (df[time_col] > cur_val_end) & (df[time_col] <= test_end)
        if test_mask.sum() == 0: break
        yield np.where(train_mask)[0], np.where(val_mask)[0], np.where(test_mask)[0]
        cur_train_end += pd.Timedelta(step)
        cur_val_end += pd.Timedelta(step)
