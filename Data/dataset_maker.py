import csv
from pathlib import Path

import numpy as np
import pandas as pd
from datasets import Dataset
from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit
from rapidfuzz import fuzz, process
from sklearn.preprocessing import MultiLabelBinarizer

from Data.paths import SPLITS_DIR, XED_DIR
from Data.util import split_labels

SPLIT_NAMES = ("train", "develop", "test")
EMOTIONS = ["Anger", "Anticipation", "Disgust", "Fear", "Joy", "Sadness", "Surprise", "Trust"]


def normalise_sentence(sentence: str) -> str:
    return "".join(str(sentence).casefold().split())


def read_tsv(path: Path) -> pd.DataFrame:
    # quoting off to prevent a quote inside a subtitle swallowing lines
    return pd.read_csv(path, sep="\t", header=None, quoting=csv.QUOTE_NONE, keep_default_na=False)


def load_labelled_frame(path: Path) -> pd.DataFrame:
    frame: pd.DataFrame = split_labels(read_tsv(path))
    frame["key"] = frame[0].map(normalise_sentence)
    return frame


def find_row(frame: pd.DataFrame, sentence: str, fuzzy_threshold: int) -> pd.Series | None:
    key: str = normalise_sentence(sentence)
    if key == "":
        return None

    exact: pd.DataFrame = frame[frame["key"] == key]
    if not exact.empty:
        return exact.iloc[0]

    match = process.extractOne(
        key, frame["key"].drop_duplicates().tolist(),
        scorer=fuzz.ratio, score_cutoff=fuzzy_threshold,
    )
    if match is None:
        return None

    return frame[frame["key"] == match[0]].iloc[0]


def build_pairs(english_frame: pd.DataFrame,
                target_frame: pd.DataFrame,
                alignment_frame: pd.DataFrame,
                fuzzy_threshold: int,
                ) -> pd.DataFrame:
    rows: list[dict] = []
    for _, pair in alignment_frame.iterrows():
        english_row = find_row(english_frame, pair[4], fuzzy_threshold)
        target_row = find_row(target_frame, pair[5], fuzzy_threshold)
        if english_row is None or target_row is None:
            continue

        rows.append({
            # file + line number of both subtitles to identify alignment
            "id": f"{pair[0]}:{pair[2]}|{pair[1]}:{pair[3]}",
            "english": english_row[0],
            "english_labels": english_row[1],
            "target": target_row[0],
            "target_labels": target_row[1],
        })

    return pd.DataFrame(rows)


def group_duplicates(pairs: pd.DataFrame) -> pd.Series:
    # pairs that share english or target sentence end up in the same group
    parent: list[int] = list(range(len(pairs)))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for column in ("english", "target"):
        first_seen: dict[str, int] = {}
        for i, sentence in enumerate(pairs[column].map(normalise_sentence)):
            if sentence in first_seen:
                parent[find(i)] = find(first_seen[sentence])
            else:
                first_seen[sentence] = i

    return pd.Series([find(i) for i in range(len(pairs))], index=pairs.index)


def split_indices(labels: np.ndarray, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    # 70% train, 30% rest
    splitter = MultilabelStratifiedShuffleSplit(n_splits=1, test_size=0.3, random_state=seed)
    train_idx, rest_idx = next(splitter.split(labels, labels))

    # rest -> 20% develop, 10% test
    rest_labels = labels[rest_idx]
    splitter = MultilabelStratifiedShuffleSplit(n_splits=1, test_size=1 / 3, random_state=seed)
    develop_in_rest, test_in_rest = next(splitter.split(rest_labels, rest_labels))

    return train_idx, rest_idx[develop_in_rest], rest_idx[test_in_rest]


def assign_splits(pairs: pd.DataFrame, seed: int) -> pd.Series:
    groups: pd.Series = group_duplicates(pairs)

    # split whole groups to prevent duplicates from ending up in different parts
    group_labels = pairs.groupby(groups)["english_labels"].agg(lambda lists: sorted({label for labels in lists for label in labels}))
    label_matrix = MultiLabelBinarizer(classes=range(1, 9)).fit_transform(group_labels)
    group_ids = group_labels.index.to_numpy()

    split_of_group: dict[int, str] = {}
    for name, indices in zip(SPLIT_NAMES, split_indices(label_matrix, seed)):
        for group in group_ids[indices]:
            split_of_group[group] = name

    return groups.map(split_of_group)


def to_dataset(pairs: pd.DataFrame, side: str) -> Dataset:
    return Dataset.from_dict({
        "id": pairs["id"].tolist(),
        "sentence": pairs[side].tolist(),
        "labels": pairs[f"{side}_labels"].tolist(),
    })


def create_split_dataset(english_path: Path,
                         target_path: Path,
                         alignment_path: Path,
                         split_path: Path,
                         seed: int = 0,
                         fuzzy_threshold: int = 90,
                         ) -> tuple[tuple[Dataset, ...], ...]:
    english_frame: pd.DataFrame = load_labelled_frame(english_path)
    target_frame: pd.DataFrame = load_labelled_frame(target_path)

    alignment_frame: pd.DataFrame = read_tsv(alignment_path)
    alignment_frame = alignment_frame[alignment_frame[0].str.startswith("en/")]

    pairs: pd.DataFrame = build_pairs(english_frame, target_frame, alignment_frame, fuzzy_threshold)

    # the saved file should be source of truth so everybody can train on the same split
    if split_path.exists():
        split_by_id = pd.read_csv(split_path, sep="\t", index_col="id")["split"]
        splits = pairs["id"].map(split_by_id)
        if splits.isna().any():
            raise ValueError(f"{split_path} does not match the pairs, delete it to make a new split")
    else:
        splits = assign_splits(pairs, seed)
        split_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({"id": pairs["id"], "split": splits}).to_csv(split_path, sep="\t", index=False)

    english_splits = tuple(to_dataset(pairs[splits == name], "english") for name in SPLIT_NAMES)
    target_splits = tuple(to_dataset(pairs[splits == name], "target") for name in SPLIT_NAMES)

    return english_splits, target_splits


# (train, develop, test) with row i being the same aligned pair
def load_dutch_splits() -> tuple[tuple[Dataset, ...], ...]:
    return create_split_dataset(
        XED_DIR / "AnnotatedData" / "en-annotated.tsv",
        XED_DIR / "Projections" / "nl-projections.tsv",
        XED_DIR / "subtitle-retrieval" / "students" / "pairs-nl.txt",
        SPLITS_DIR / "nl.tsv",
    )


if __name__ == "__main__":
    english_splits, dutch_splits = load_dutch_splits()
    for name, dataset in zip(SPLIT_NAMES, english_splits):
        print(name, len(dataset))
