from collections.abc import Callable
from pathlib import Path

import pandas as pd
from datasets import Dataset


def create_split_dataset(data_path: Path, 
                         seed: int = 0, 
                         preprocesses: list[Callable[[pd.DataFrame], pd.DataFrame]] | None = None,
                         postprocesses: list[Callable[[Dataset], Dataset]] | None = None) -> tuple[Dataset, ...]:
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

    return_value = [train, develop, test]

    if postprocesses != None:
        for index, data in enumerate(return_value):
            for process in postprocesses:
                return_value[index] = process(return_value[index])
    
    return tuple(return_value)


if __name__ == "__main__":
    """Test Code"""
    from Data.postprocesses import *
    from Data.preprocesses import *
    
    test_data_path = Path("./XED/AnnotatedData/en-annotated.tsv")
    for i in create_split_dataset(test_data_path, 
                                  preprocesses=[split_labels],
                                  postprocesses=[split_allignment])[0]:
        print(i)

    delete_dupli_leak(create_split_dataset(test_data_path))
    