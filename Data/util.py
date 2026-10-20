import pandas as pd
from datasets import Dataset


def split_labels(data: pd.DataFrame) -> pd.DataFrame:
    data[1] = data[1].apply(lambda labels: [int(label) for label in labels.split(",")])
    return data


def split_allignment(data: Dataset) -> Dataset:
    rows: list[tuple[str, int]] = []
    for sentence, labels in zip(data["sentence"], data["labels"]):
        for label in labels:
            rows.append((sentence, label))

    return Dataset.from_pandas(pd.DataFrame(rows, columns=["sentence", "label"]))
