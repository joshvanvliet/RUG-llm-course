from collections.abc import Callable
from pathlib import Path

import pandas as pd
from datasets import Dataset


def create_split_dataset(data_path: Path, 
                         seed: int = 0, 
                         preprocesses: list[Callable[[pd.DataFrame], pd.DataFrame]] | None = None,
                         postprocesses: list[Callable[[Dataset], Dataset]] | None = None) -> tuple[Dataset, Dataset, Dataset]:
    pandas_data: pd.DataFrame = pd.DataFrame(pd.read_csv(data_path, sep="\t", header=None))
    
    if preprocesses != None:
        for process in preprocesses:
            pandas_data = process(pandas_data)

    dataset = Dataset.from_pandas(pandas_data)
    
    dataset = dataset.shuffle(seed)
    temp = dataset.train_test_split(seed=seed, test_size=0.3)
    train = temp["train"]
    rest = temp["test"]
    
    temp = rest.train_test_split(seed=seed, test_size=2/3)
    develop = temp["test"]
    test = temp["train"]

    if postprocesses != None:
        for data in (train, develop, test):
            for process in postprocesses:
                data = process(data)
    
    return (train, develop, test)


if __name__ == "__main__":
    from Data.preprocesses import *
    
    test_data_path = Path("./XED/AnnotatedData/en-annotated.tsv")
    for i in create_split_dataset(test_data_path, preprocesses=[split_labels])[0]:
        print(i)
    