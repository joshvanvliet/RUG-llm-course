import pandas as pd
from datasets import Dataset


def split_allignment(data: Dataset) -> Dataset:
    panda: pd.DataFrame = data.to_pandas()
    newpandas: list[tuple[str, int]] = []

    for index, row in panda.iterrows():
        if len(row["1"]) == 1:
            newpandas.append((row["0"], row["1"][0]))
        else:
            for label in row["1"]:
                newpandas.append((row["0"], label))
    
    return Dataset.from_pandas(pd.DataFrame(newpandas))