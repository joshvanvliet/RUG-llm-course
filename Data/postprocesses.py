from collections import Counter

import pandas as pd
from datasets import Dataset


def split_allignment(data: Dataset) -> Dataset:
    panda = data.to_pandas()
    newpandas: list[tuple[str, int]] = []

    for index, row in panda.iterrows():
        if len(row["1"]) == 1:
            newpandas.append((row["0"], row["1"][0]))
        else:
            for label in row["1"]:
                newpandas.append((row["0"], label))

    return Dataset.from_pandas(pd.DataFrame(newpandas))

def delete_dupli_leak(datasets: tuple[Dataset, ...]) -> tuple[Dataset,...]:
    duplicate_counter = Counter()

    for dataset in datasets:
        duplicate_counter += Counter(set(dataset["0"]))

    duplicate_sentences: set[str] = set()
    for (sentence, count) in duplicate_counter.items():
        if count >= 2:
            duplicate_sentences.add(sentence)

    return_value: list[Dataset] = []

    for dataset in datasets:
        return_value.append(dataset.select([
            index for index, sentence in enumerate(dataset["0"]) 
            if sentence not in duplicate_sentences
        ]))

    return tuple(return_value)
