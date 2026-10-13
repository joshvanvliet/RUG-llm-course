from pathlib import Path

import pandas as pd
from datasets import Dataset
from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit
from sklearn.preprocessing import MultiLabelBinarizer

from Data.util import split_labels


def create_split_dataset(data_path: Path, 
                         seed: int = 0, 
                        ) -> tuple[Dataset, ...]:
    pandas_data: pd.DataFrame = split_labels(pd.DataFrame(pd.read_csv(data_path, sep="\t", header=None)))
    data = Dataset.from_pandas(pandas_data)
    
    y = MultiLabelBinarizer(classes=range(1, 9)).fit_transform(data["1"])
    
    first = MultilabelStratifiedShuffleSplit(
        n_splits=1, test_size=0.3, random_state=seed
    )
    train_idx, rest_idx = next(first.split(y, y))
    
    second = MultilabelStratifiedShuffleSplit(
        n_splits=1, test_size=(1/3), random_state=seed
    )
    rest_y = y[rest_idx]
    dev_relative, test_relative = next(second.split(rest_y, rest_y))

    train = data.select(train_idx.tolist())
    develop = data.select(rest_idx[dev_relative].tolist())
    test = data.select(rest_idx[test_relative].tolist())
    
    return train, develop, test


if __name__ == "__main__":
    """Test Code"""
    test_data_path = Path("./XED/AnnotatedData/en-annotated.tsv")
    t, d, te = create_split_dataset(test_data_path)
    print(t.to_pandas())
    