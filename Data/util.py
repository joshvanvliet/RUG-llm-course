from collections import Counter

import pandas as pd
from datasets import Dataset


def split_labels(data: pd.DataFrame) -> pd.DataFrame:
    data[1] = data[1].apply(lambda labels: [int(label) for label in labels.split(",")])
    return data


def split_allignment(data: Dataset) -> Dataset:
    rows: list[tuple[str, int]] = []
    for sentence, labels in zip(data["0"], data["1"]):
        for label in labels:
            rows.append((sentence, label))

    return Dataset.from_pandas(pd.DataFrame(rows))


def delete_dupli_leak(datasets: tuple[Dataset, ...]) -> tuple[Dataset, ...]:
    counts: Counter = Counter()
    for dataset in datasets:
        counts += Counter(set(dataset["0"]))

    duplicates: set[str] = {sentence for sentence, count in counts.items() if count >= 2}

    cleaned: list[Dataset] = []
    for dataset in datasets:
        keep: list[int] = [i for i, sentence in enumerate(dataset["0"]) if sentence not in duplicates]
        cleaned.append(dataset.select(keep))

    return tuple(cleaned)
