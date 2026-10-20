from pathlib import Path

import pandas as pd
from datasets import Dataset
from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit
from rapidfuzz import fuzz, process
from sklearn.preprocessing import MultiLabelBinarizer

from Data.paths import XED_DIR
from Data.util import split_labels


def normalise_sentence(sentence: str) -> str:
    return " ".join(str(sentence).casefold().split())


def find_sentence(frame: pd.DataFrame,
                  sentence: str,
                  fuzzy_threshold: int,
                  ) -> pd.DataFrame:
    wanted: str = normalise_sentence(sentence)
    normalised: pd.Series = frame[0].map(normalise_sentence)

    exact: pd.DataFrame = frame[normalised == wanted]
    if not exact.empty:
        return exact

    match = process.extractOne(
        wanted, normalised.drop_duplicates().tolist(),
        scorer=fuzz.ratio, score_cutoff=fuzzy_threshold,
    )
    if match is None:
        return frame.iloc[0:0]

    return frame[normalised == match[0]]


def split_dataset(data: Dataset, seed: int) -> tuple[Dataset, Dataset, Dataset]:
    labels = MultiLabelBinarizer(classes=range(1, 9)).fit_transform(data["1"])

    # 70% train, 30% rest
    splitter = MultilabelStratifiedShuffleSplit(n_splits=1, test_size=0.3, random_state=seed)
    train_idx, rest_idx = next(splitter.split(labels, labels))

    # rest -> 20% develop, 10% test
    rest_labels = labels[rest_idx]
    splitter = MultilabelStratifiedShuffleSplit(n_splits=1, test_size=1 / 3, random_state=seed)
    develop_idx, test_idx = next(splitter.split(rest_labels, rest_labels))

    return (
        data.select(train_idx.tolist()),
        data.select(rest_idx[develop_idx].tolist()),
        data.select(rest_idx[test_idx].tolist()),
    )


def create_split_dataset(english_path: Path,
                         target_path: Path,
                         alignment_path: Path,
                         seed: int = 0,
                         fuzzy_threshold: int = 90,
                         ) -> tuple[tuple[Dataset, ...], ...]:
    english_frame: pd.DataFrame = split_labels(pd.read_csv(english_path, sep="\t", header=None))
    target_frame: pd.DataFrame = split_labels(pd.read_csv(target_path, sep="\t", header=None))

    alignment_frame: pd.DataFrame = pd.read_csv(alignment_path, sep="\t", header=None)
    alignment_frame = alignment_frame[alignment_frame[0].str.contains("en/", na=False)]

    english_rows: list[list] = []
    target_rows: list[list] = []
    for _, row in alignment_frame.iterrows():
        english_matches: pd.DataFrame = find_sentence(english_frame, row[4], fuzzy_threshold)
        target_matches: pd.DataFrame = find_sentence(target_frame, row[5], fuzzy_threshold)
        if english_matches.empty or target_matches.empty:
            continue
        english_rows += english_matches.values.tolist()
        target_rows += target_matches.values.tolist()

    english: Dataset = Dataset.from_pandas(pd.DataFrame(english_rows))
    target: Dataset = Dataset.from_pandas(pd.DataFrame(target_rows))

    return split_dataset(english, seed), split_dataset(target, seed)


if __name__ == "__main__":
    english_path = XED_DIR / "AnnotatedData" / "en-annotated.tsv"
    dutch_path = XED_DIR / "Projections" / "nl-projections.tsv"
    alignment_path = XED_DIR / "subtitle-retrieval" / "students" / "pairs-nl.txt"
    english_splits, dutch_splits = create_split_dataset(
        english_path, dutch_path, alignment_path
    )
