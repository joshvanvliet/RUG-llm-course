import pandas as pd


def split_labels(data: pd.DataFrame) -> pd.DataFrame:
    data[1] = data[1].apply(lambda x : [int(_) for _ in x.split(",")])
    
    return data

    