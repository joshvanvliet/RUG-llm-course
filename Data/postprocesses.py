import pandas as pd
from datasets import Dataset
from collections import Counter


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

def delete_dupli_leak(datasets: tuple[Dataset, Dataset, Dataset]) -> tuple[Dataset, Dataset, Dataset]:
    duplicate_counter = Counter()

    for dataset in datasets:
        duplicate_counter += Counter(set(dataset["0"]))

    duplicate_sentences: list[str] = []
    for (sentence, count) in duplicate_counter.items():
        if count >= 2:
            duplicate_sentences.append(sentence)

    return_value: list[pd.DataFrame] = []

    for dataset in datasets:
        return_value.append(dataset.to_pandas().)