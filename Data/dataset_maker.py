from pathlib import Path

from datasets import Dataset
import pandas as pd


def create_split_dataset(data_path:Path):
    dataframe = pd.DataFrame(pd.read_csv(data_path, sep="\t", header=None))
    dataset = Dataset.from_pandas(dataframe)

    print(dataset[0:10])


if __name__ == "__main__":
    test_data_path = Path("./XED/AnnotatedData/en-annotated.tsv")
    create_split_dataset(test_data_path)
