from pathlib import Path

import pandas as pd
from datasets import Dataset
from sklearn.model_selection import train_test_split

from Data.util import split_labels


def create_split_dataset(data_path: Path, 
                         seed: int = 0, 
                        ) -> tuple[Dataset, ...]:
    pandas_data: pd.DataFrame = split_labels(pd.DataFrame(pd.read_csv(data_path, sep="\t", header=None)))

    newpandas: list[tuple[str, int]] = []

    for index, row in pandas_data.iterrows():
        if len(row[1]) == 1:
            newpandas.append((row[0], row[1][0]))
        else:
            for label in row[1]:
                newpandas.append((row[0], label))

    pandas_data = pd.DataFrame(newpandas)

    train_frame, rest_frame = train_test_split(pandas_data, train_size=0.7, stratify=pandas_data[1], random_state=seed)

    dev_frame, test_frame = train_test_split(rest_frame, train_size=(2/3), stratify=rest_frame[1], random_state=seed)

    return tuple([Dataset.from_pandas(_) for _ in [train_frame, dev_frame, test_frame]])


if __name__ == "__main__":
    """Test Code"""
    test_data_path = Path("./XED/AnnotatedData/en-annotated.tsv")
    create_split_dataset(test_data_path)
    