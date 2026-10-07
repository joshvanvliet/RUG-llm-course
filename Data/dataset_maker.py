from collections.abc import Callable
from pathlib import Path

import pandas as pd
from datasets import Dataset


def create_split_dataset(data_path: Path, seed: int = 0, preprocesses: list[Callable[[Dataset], Dataset]] | None = None) -> tuple[Dataset, Dataset, Dataset]:
    dataset = Dataset.from_pandas(pd.DataFrame(pd.read_csv(data_path, sep="\t", header=None)))
    
    if preprocesses != None:
        for process in preprocesses:
            dataset = process(dataset)
    
    dataset = dataset.shuffle(seed)
    temp = dataset.train_test_split(seed=seed, test_size=0.3)
    train = temp["train"]
    rest = temp["test"]
    
    temp = rest.train_test_split(seed=seed, test_size=2/3)
    develop = temp["test"]
    test = temp["train"]
    
    return (train, develop, test)


if __name__ == "__main__":
    import pprint
    test_data_path = Path("./XED/AnnotatedData/en-annotated.tsv")
    pprint.pprint(create_split_dataset(test_data_path))
    for i in create_split_dataset(test_data_path)[0]:
        print(i)
    